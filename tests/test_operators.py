"""Offline unit tests (no LLM/API needed).

Covers: text_clean, chunker, triple_validator, the DAG executor's topological
ordering, and local graph CRUD. The LLM-backed operators (NER/RE) are exercised
in the demos with a real API key, not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medigraph.operators.text_clean import TextCleanOperator
from medigraph.operators.chunker import ChunkerOperator
from medigraph.operators.triple_validator import TripleValidatorOperator
from medigraph.agents.dag_executor import topological_order
from medigraph.graph.local_store import LocalGraphStore
from medigraph.graph.vector_store import LocalVectorStore
from medigraph.schema.normalize import canonical_key, is_structural_noise, is_valid_entity_name


def test_text_clean_removes_links_and_noise():
    op = TextCleanOperator()
    raw = "# 标题\n\nAdvertisement\n[click](http://x.com) 正常内容句子。\n![img](a.png)\n"
    out = op.run({"text": raw})["text"]
    assert "Advertisement" not in out
    assert "http://x.com" not in out
    assert "# 标题" in out
    assert "正常内容句子" in out


def test_chunker_splits_by_heading():
    op = ChunkerOperator(max_chars=50)
    text = "# A\n" + "甲" * 40 + "\n\n# B\n" + "乙" * 40
    chunks = op.run({"text": text})["chunks"]
    assert len(chunks) >= 2


def test_triple_validator_schema_and_conflict():
    op = TripleValidatorOperator(min_confidence=0.5)
    triples = [
        {"head": "嗜铬细胞瘤", "head_type": "Tumor", "relation": "positive_marker",
         "tail": "chromogranin", "tail_type": "Biomarker", "confidence": 0.9},
        # schema violation: recommend_drug tail must be Drug
        {"head": "嗜铬细胞瘤", "head_type": "Tumor", "relation": "recommend_drug",
         "tail": "头痛", "tail_type": "Symptom", "confidence": 0.9},
        # low confidence
        {"head": "嗜铬细胞瘤", "head_type": "Tumor", "relation": "has_symptom",
         "tail": "出汗", "tail_type": "Symptom", "confidence": 0.1},
    ]
    res = op.run({"triples": triples})
    assert len(res["valid"]) == 1
    assert res["valid"][0]["tail"] == "chromogranin"
    assert len(res["rejected"]) == 2


def test_topological_order():
    dag = [
        {"id": "c", "op": "x", "deps": ["b"]},
        {"id": "b", "op": "x", "deps": ["a"]},
        {"id": "a", "op": "x", "deps": []},
    ]
    order = topological_order(dag)
    assert order.index("a") < order.index("b") < order.index("c")


def test_local_graph_crud_and_neighbors():
    g = LocalGraphStore()
    g.upsert_triple("嗜铬细胞瘤", "Tumor", "positive_marker", "S100", "Biomarker", 0.9, "doc1")
    g.upsert_triple("嗜铬细胞瘤", "Tumor", "associated_gene", "RET", "Gene", 0.8, "doc1")
    stats = g.stats()
    assert stats["num_entities"] == 3
    assert stats["num_triples"] == 2
    resolved = g.find_entities(["嗜铬细胞瘤"])
    assert "嗜铬细胞瘤" in resolved
    nbrs = g.neighbors("嗜铬细胞瘤", hops=1)
    assert len(nbrs) == 2


def test_entity_normalization_and_noise():
    # case/space variants share a canonical key
    assert canonical_key("Pheochromocytoma") == canonical_key("pheochromocytoma ")
    # section headings are flagged as structural noise, real entities are not
    assert is_structural_noise("Radiology images")
    assert is_structural_noise("Gross description")
    assert not is_structural_noise("pheochromocytoma")


def test_placeholder_entity_filtering():
    assert not is_valid_entity_name("Ⅰ")
    assert not is_valid_entity_name("|")
    assert is_valid_entity_name("硝苯地平缓释片Ⅰ")

    op = TripleValidatorOperator(min_confidence=0.5)
    res = op.run(
        {
            "triples": [
                {
                    "head": "高血压",
                    "head_type": "Disease",
                    "relation": "recommend_drug",
                    "tail": "Ⅰ",
                    "tail_type": "Drug",
                    "confidence": 1.0,
                },
                {
                    "head": "高血压",
                    "head_type": "Disease",
                    "relation": "recommend_drug",
                    "tail": "硝苯地平缓释片Ⅰ",
                    "tail_type": "Drug",
                    "confidence": 1.0,
                },
            ]
        }
    )
    assert [t["tail"] for t in res["valid"]] == ["硝苯地平缓释片Ⅰ"]


def test_graph_merges_case_variants():
    g = LocalGraphStore()
    g.upsert_triple("Pheochromocytoma", "Tumor", "positive_marker", "S100", "Biomarker", 0.9, "d")
    g.upsert_triple("pheochromocytoma", "Tumor", "associated_gene", "RET", "Gene", 0.8, "d")
    # "Pheochromocytoma" and "pheochromocytoma" must collapse to one node
    assert g.stats()["num_entities"] == 3


def test_vector_store_search():
    vs = LocalVectorStore()
    vs.add(
        ["disease A info", "drug B info"],
        ["doc1", "doc2"],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    hits = vs.search([0.9, 0.1], k=1)
    assert len(hits) == 1
    assert hits[0]["source"] == "doc1"


# ---- Task 3 (offline) ---- #

def test_relational_build_and_reuse(tmp_path):
    import sqlite3
    from medigraph.analysis.graph_profile import load_graph
    from medigraph.analysis.relational import build_db, derive_vocab

    store, _ = load_graph(None)  # embedded example graph
    vocab = derive_vocab(store)
    assert "高血压" in vocab and vocab["高血压"]["department"]  # reuse from KG triples
    db = tmp_path / "t.db"
    summary = build_db(db, store, n_visits=100, seed=1)
    assert summary["n_visits"] == 100
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM patient_visits").fetchone()[0]
    assert n == 100
    # KG mirror tables populated (traceable reuse)
    assert conn.execute("SELECT COUNT(*) FROM kg_triples").fetchone()[0] > 0
    conn.close()


def test_derive_vocab_curates_thin_disease_pools():
    """Regression: literal-filter NL2SQL questions must get real, non-empty rows.

    Spreading N visits across every KG disease with *any* department/drug edge
    (the CM3KG import has ~1,200) leaves most diseases with 0-1 visits: a
    question like "高血压患者..." then returns an empty result on both the gold
    and the predicted SQL, so it "passes" by vacuous empty==empty agreement
    without validating that the predicted SQL is actually correct. Curating to
    the richest `max_diseases` entries (plus a forced anchor set already
    referenced by the NL2SQL few-shot pool / gold questions / stress generator)
    keeps literal filters testable. See relational.derive_vocab's docstring.
    """
    from medigraph.graph.local_store import LocalGraphStore
    from medigraph.analysis.relational import ANCHOR_DISEASES, derive_vocab, generate_rows

    store = LocalGraphStore()
    # One well-documented anchor disease (many drugs -- would rank in the top
    # slice on richness alone) plus 200 thin ones that would dilute a naive
    # uniform draw across "every disease with an edge".
    for index, drug in enumerate(f"drug_{i}" for i in range(6)):
        store.upsert_triple("高血压", "Disease", "recommend_drug", drug, "Drug")
    store.upsert_triple("高血压", "Disease", "treated_in_department", "内科", "Department")
    for index in range(200):
        store.upsert_triple(f"稀有病_{index}", "Disease", "recommend_drug", "某药", "Drug")

    vocab = derive_vocab(store, max_diseases=20)
    assert len(vocab) <= 20 + len(ANCHOR_DISEASES)
    assert "高血压" in vocab, "richest disease must survive curation"

    rows = generate_rows(store, n_visits=300, seed=1, max_diseases=20)
    sampled = {visit[4] for visit in rows["patient_visits"]}
    assert "高血压" in sampled, (
        "curated vocab must be small enough that a 300-visit draw actually "
        "samples the anchor disease, not just includes it in the candidate pool"
    )
    hypertension_visits = sum(1 for visit in rows["patient_visits"] if visit[4] == "高血压")
    assert hypertension_visits >= 3, "anchor disease should get a real, countable sample"


def test_derive_vocab_anchors_survive_even_when_not_richest():
    """Anchor diseases referenced by name elsewhere in the codebase (few-shot
    pool, hand-written gold, stress generator) must never be curated away, even
    if they happen to rank outside the richness cut."""
    from medigraph.graph.local_store import LocalGraphStore
    from medigraph.analysis.relational import derive_vocab

    store = LocalGraphStore()
    # Anchor with only the minimum required edge (department, no drugs) --
    # deliberately the least "rich" possible entry.
    store.upsert_triple("高血压", "Disease", "treated_in_department", "内科", "Department")
    for index in range(50):
        for drug in (f"drug_{index}_{j}" for j in range(10)):
            store.upsert_triple(f"富病_{index}", "Disease", "recommend_drug", drug, "Drug")

    vocab = derive_vocab(store, max_diseases=10)
    assert "高血压" in vocab, "anchor must be force-included despite low richness rank"


def test_nl2sql_guards():
    from medigraph.analysis.nl2sql import NL2SQL
    assert NL2SQL._is_readonly("SELECT * FROM patient_visits")
    assert not NL2SQL._is_readonly("DELETE FROM patient_visits")
    assert NL2SQL._extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_chart_type_picker():
    from medigraph.analysis import viz
    assert viz.pick_chart_type("2024年每月就诊量趋势", ["m", "c"], [("2024-01", 5)]) == "line"
    assert viz.pick_chart_type("男女就诊比例", ["g", "c"], [("男", 5)]) == "pie"
    assert viz.pick_chart_type("各科室就诊量", ["d", "c"], [("心内科", 5)]) == "bar"
