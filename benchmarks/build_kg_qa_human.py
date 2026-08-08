r"""Build a human-authored, grounded QA set for Ragas evaluation of GraphRAG.

Why this exists
----------------
`benchmarks/eval_kg_qa.py` generates its questions *and* answers them using the
same graph-traversal code (`LocalGraphStore.neighbors`/`traverse_paths`), so its
1.000/1.000/1.000 is a self-consistency check on the retrieval primitive, not an
independent measurement of answer quality -- it never calls the LLM composition
step `QAAgent.answer()` actually uses in production. See the "读取而非自证"
note in `docs/EVIDENCE_MAP.md` for the reclassification this file supports.

This script builds a *different* kind of question: every reference answer and
reference context is extracted directly from `graph_scaled.json` (the self-
produced Task-2 graph, not the imported CM3KG baseline) and verified against the
loaded graph at generation time, so nothing here can drift from the actual data.
But the *question wording* is natural Chinese prose across ten relation
categories, one/two/three-hop chains, and out-of-graph negatives -- exercising
QAAgent's real pipeline (NER anchor resolution -> subgraph retrieval -> LLM
composition) end to end. `benchmarks/eval_ragas_kg_qa.py` runs each question
through the real agent and scores the real answer with Ragas.

    python benchmarks/build_kg_qa_human.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.agents.qa_agent import _relation_label  # noqa: E402
from medigraph.graph.local_store import LocalGraphStore  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

OUT = Path(__file__).resolve().parent / "kg_qa_human.json"
GRAPH = Path(OUTPUTS_DIR) / "graph_scaled.json"

random.seed(20260731)

# relation -> (question verb phrase, answer verb phrase, min targets required)
RELATION_TEMPLATES: dict[str, tuple[str, str, int]] = {
    "cmeie:clinical_manifestation": ("的常见临床表现有哪些", "的常见临床表现包括", 3),
    "cmeie:drug_treatment": ("常用的治疗药物有哪些", "常用的治疗药物包括", 3),
    "cmeie:complication": ("常见的并发症有哪些", "常见的并发症包括", 2),
    "cmeie:related_cause": ("可能与哪些情况相关", "可能与以下情况相关：", 3),
    "cmeie:etiology": ("常见的病因有哪些", "常见的病因包括", 3),
    "cmeie:differential_diagnosis": ("需要与哪些疾病做鉴别诊断", "需要与以下疾病鉴别：", 2),
    "cmeie:pathological_classification": ("有哪些病理分型", "的病理分型包括", 2),
    "cmeie:high_risk_factor": ("的高危因素有哪些", "的高危因素包括", 2),
    "cmeie:adjuvant_therapy": ("有哪些辅助或非药物治疗方式", "的辅助治疗方式包括", 2),
    "cmeie:laboratory_examination": ("通常需要做哪些实验室检查", "通常需要的实验室检查包括", 2),
}

ANCHOR_DISEASES = (
    "糖尿病", "高血压", "冠心病", "脑炎", "急性胰腺炎",
    "非小细胞肺癌", "HIV感染", "慢性胰腺炎", "登革热", "疟疾感染",
)

NEGATIVE_QUESTIONS = (
    "幻想性紫罗兰综合征的常见并发症有哪些？",
    "第九型星际眩晕症通常需要做哪些检查？",
    "蓝焰热病的推荐治疗药物是什么？",
    "零号漂浮症的病理分型有哪些？",
    "逆光性骨骼软化综合征的高危因素有哪些？",
    "幽冥回响病的病因是什么？",
)


#: Entities longer than this are almost never a clean noun phrase in this
#: domain -- they are typically a raw extraction span that swallowed a whole
#: clinical-criteria sentence or a stray statistics fragment (both observed in
#: graph_scaled.json, e.g. "近年，各地CKD流行病学调查成年人CKD的患病率存在
#: 一定差异，为9%~14%" surfacing as a `related_cause` target). CMeIE entities
#: can legitimately run long (clinical criteria phrases), so this is a coarse
#: filter, not a claim that everything under the cap is clean -- it exists to
#: keep generated questions/references readable, not to certify graph quality.
_MAX_ENTITY_CHARS = 25


def _is_reasonable_entity(name: str, head: str) -> bool:
    if not name or name == head:  # self-loops are extraction noise, not entities
        return False
    if len(name) > _MAX_ENTITY_CHARS:
        return False
    if any(mark in name for mark in "。；：，、"):  # sentence-internal punctuation
        return False
    return True


def _clean_targets(store: LocalGraphStore, head: str, relation_filter=None) -> dict[str, list[str]]:
    """head's out-edges grouped by relation, entity-noise filtered.

    Shorter survivors sort first within each relation, so short/legitimate entity
    names win over long-but-under-the-cap ones when a question only samples the
    top few -- readability over exhaustiveness for a generated eval set.
    """
    by_relation: dict[str, list[str]] = {}
    for _, tail, data in store.g.out_edges(head, data=True):
        relation = data.get("relation", "")
        if relation_filter is not None and relation not in relation_filter:
            continue
        if not _is_reasonable_entity(tail, head):
            continue
        by_relation.setdefault(relation, []).append(tail)
    for relation, targets in by_relation.items():
        by_relation[relation] = sorted(set(targets), key=lambda t: (len(t), t))
    return by_relation


def _triple_text(head: str, relation: str, tail: str) -> str:
    return f"{head} --[{_relation_label({'relation': relation})}]--> {tail}"


def build_one_hop(store: LocalGraphStore, n_target: int) -> list[dict]:
    samples: list[dict] = []
    for anchor in ANCHOR_DISEASES:
        if anchor not in store.g.nodes:
            continue
        by_relation = _clean_targets(store, anchor)
        # Two categories per anchor, picked deterministically by richness, so the
        # set spans relation types instead of always picking the same one.
        candidates = [
            (relation, targets)
            for relation, targets in by_relation.items()
            if relation in RELATION_TEMPLATES and len(targets) >= RELATION_TEMPLATES[relation][2]
        ]
        candidates.sort(key=lambda item: -len(item[1]))
        for relation, targets in candidates[:2]:
            question_phrase, answer_phrase, _ = RELATION_TEMPLATES[relation]
            top = targets[:5]
            samples.append(
                {
                    "question": f"{anchor}{question_phrase}？",
                    "category": relation,
                    "hops": 1,
                    "anchor": anchor,
                    "gold_entities": top,
                    "reference": f"{anchor}{answer_phrase}{'、'.join(top)}。",
                    "reference_context": [_triple_text(anchor, relation, t) for t in top],
                }
            )
            if len(samples) >= n_target:
                return samples
    return samples


def build_two_hop(store: LocalGraphStore, n_target: int) -> list[dict]:
    samples: list[dict] = []
    for anchor in ANCHOR_DISEASES:
        if anchor not in store.g.nodes or len(samples) >= n_target:
            continue
        # anchor -[complication or related_cause]-> mid -[clinical_manifestation or
        # drug_treatment]-> targets. The question names the intermediate entity, so
        # a correct answer genuinely requires chaining through it, not just
        # re-reading the anchor's direct edges.
        first_hop = [
            (tail, data.get("relation"))
            for _, tail, data in store.g.out_edges(anchor, data=True)
            if data.get("relation") in ("cmeie:complication", "cmeie:related_cause")
            and _is_reasonable_entity(tail, anchor)
        ]
        random.shuffle(first_hop)
        for mid, first_relation in first_hop:
            if mid not in store.g.nodes:
                continue
            second_hop = _clean_targets(
                store, mid, relation_filter={"cmeie:clinical_manifestation", "cmeie:drug_treatment"}
            )
            for relation in ("cmeie:clinical_manifestation", "cmeie:drug_treatment"):
                targets = second_hop.get(relation, [])
                if len(targets) < 2:
                    continue
                question_phrase, answer_phrase, _ = RELATION_TEMPLATES[relation]
                top = targets[:4]
                first_label = _relation_label({"relation": first_relation})
                samples.append(
                    {
                        "question": f"{anchor}的{first_label}之一是{mid}，{mid}{question_phrase}？",
                        "category": f"{first_relation}+{relation}",
                        "hops": 2,
                        "anchor": anchor,
                        "gold_entities": top,
                        "reference": f"{mid}{answer_phrase}{'、'.join(top)}。",
                        "reference_context": [
                            _triple_text(anchor, first_relation, mid),
                            *[_triple_text(mid, relation, t) for t in top],
                        ],
                    }
                )
                break  # one two-hop question per (anchor, mid) pair is enough
            if len(samples) >= n_target:
                break
    return samples[:n_target]


def build_three_hop(store: LocalGraphStore, n_target: int) -> list[dict]:
    samples: list[dict] = []
    for anchor in ANCHOR_DISEASES:
        if len(samples) >= n_target:
            break
        firsts = [
            (tail, data.get("relation"))
            for _, tail, data in store.g.out_edges(anchor, data=True)
            if data.get("relation") == "cmeie:complication" and _is_reasonable_entity(tail, anchor)
        ]
        for mid1, rel1 in firsts:
            if mid1 not in store.g.nodes:
                continue
            seconds = [
                (tail, data.get("relation"))
                for _, tail, data in store.g.out_edges(mid1, data=True)
                if data.get("relation") in ("cmeie:complication", "cmeie:related_cause")
                and tail != anchor
                and _is_reasonable_entity(tail, mid1)
            ]
            for mid2, rel2 in seconds:
                if mid2 not in store.g.nodes or mid2 == mid1:
                    continue
                thirds = _clean_targets(store, mid2, relation_filter={"cmeie:clinical_manifestation"})
                targets = thirds.get("cmeie:clinical_manifestation", [])
                if len(targets) < 2:
                    continue
                top = targets[:3]
                rel1_label = _relation_label({"relation": rel1})
                rel2_label = _relation_label({"relation": rel2})
                samples.append(
                    {
                        "question": (
                            f"{anchor}的{rel1_label}之一是{mid1}，"
                            f"{mid1}的{rel2_label}之一是{mid2}，"
                            f"{mid2}的常见临床表现有哪些？"
                        ),
                        "category": "3hop_complication_chain",
                        "hops": 3,
                        "anchor": anchor,
                        "gold_entities": top,
                        "reference": f"{mid2}的常见临床表现包括{'、'.join(top)}。",
                        "reference_context": [
                            _triple_text(anchor, rel1, mid1),
                            _triple_text(mid1, rel2, mid2),
                            *[_triple_text(mid2, "cmeie:clinical_manifestation", t) for t in top],
                        ],
                    }
                )
                if len(samples) >= n_target:
                    return samples
    return samples


def build_negatives() -> list[dict]:
    return [
        {
            "question": question,
            "category": "safe_rejection",
            "hops": 0,
            "anchor": None,
            "gold_entities": [],
            "reference": "图谱中未收录该实体，应明确说明证据不足，不得编造答案。",
            "reference_context": [],
        }
        for question in NEGATIVE_QUESTIONS
    ]


def main() -> None:
    enable_utf8()
    if not GRAPH.exists():
        raise SystemExit(f"{GRAPH} not found; build it first (scripts/build_scaled_kg.py)")
    store = LocalGraphStore.load_json(GRAPH)

    one_hop = build_one_hop(store, n_target=22)
    two_hop = build_two_hop(store, n_target=10)
    three_hop = build_three_hop(store, n_target=4)
    negatives = build_negatives()

    samples = one_hop + two_hop + three_hop + negatives
    questions = [s["question"] for s in samples]
    assert len(questions) == len(set(questions)), "duplicate question text"

    report = {
        "description": (
            "Human/grounded QA set for Ragas evaluation of QAAgent (see module "
            "docstring). References and contexts are extracted from and verified "
            "against graph_scaled.json at generation time -- not hand-typed."
        ),
        "graph_file": str(GRAPH),
        "counts": {"1hop": len(one_hop), "2hop": len(two_hop), "3hop": len(three_hop), "negative": len(negatives)},
        "samples": samples,
    }
    path = write_json(report, OUT)
    print(f"built {len(samples)} questions ({report['counts']}) -> {path}")


if __name__ == "__main__":
    main()
