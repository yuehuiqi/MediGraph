"""Offline tests for ingestion, recovery, graph governance and safe analytics."""
from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from medigraph.agents.dag_executor import DAGExecutor, NodeStatus, topological_order
from medigraph.agents.qa_agent import QAAgent
from medigraph.analysis.nl2sql import NL2SQL
from medigraph.analysis.relational import SCHEMA
from medigraph.analysis.viz import pick_chart_type
from medigraph.graph.local_store import LocalGraphStore
from medigraph.operators.base import BaseOperator, OperatorMeta, register
from medigraph.operators.data_quality import DataQualityOperator
from medigraph.operators.document_loader import DocumentLoaderOperator
from medigraph.operators.pii_redact import PIIRedactOperator


class _NoLLM:
    class _Stats:
        @staticmethod
        def summary():
            return {"calls": 0}

    stats = _Stats()

    def chat(self, *args, **kwargs):
        raise AssertionError("deterministic path should not call LLM")

    def chat_json(self, *args, **kwargs):
        return {"entities": [{"name": "高血压", "type": "Disease", "confidence": 0.99}]}

    @staticmethod
    def embed(texts):
        return []


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("电话13800138000", "phone"),
        ("身份证11010519491231002X", "id_card"),
        ("邮箱doctor@example.com", "email"),
        ("卡号6222021234567890123", "bank_card"),
        ("病历号:A12345", "patient_id"),
    ],
)
def test_pii_redaction_categories(text, category):
    result = PIIRedactOperator().run({"text": text})
    assert result["redaction_report"]["by_type"][category] == 1
    assert text.split(":", 1)[-1] not in result["text"]


def _write_docx(path: Path) -> None:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>DOCX医疗文本</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


@pytest.mark.parametrize("suffix", [".txt", ".md", ".html", ".csv", ".json", ".jsonl", ".docx"])
def test_document_loader_formats(tmp_path, suffix):
    path = tmp_path / f"sample{suffix}"
    if suffix == ".html":
        path.write_text("<p>医疗文本</p><script>bad()</script>", encoding="utf-8")
    elif suffix == ".csv":
        path.write_text("疾病,药物\n高血压,氨氯地平\n", encoding="utf-8")
    elif suffix == ".json":
        path.write_text('{"疾病":"高血压"}', encoding="utf-8")
    elif suffix == ".jsonl":
        path.write_text('{"疾病":"高血压"}\n', encoding="utf-8")
    elif suffix == ".docx":
        _write_docx(path)
    else:
        path.write_text("医疗文本", encoding="utf-8")
    result = DocumentLoaderOperator().run({"path": str(path)})
    assert len(result["documents"]) == 1
    assert result["text"]
    assert not result["errors"]


def test_document_loader_reports_unsupported(tmp_path):
    path = tmp_path / "x.exe"
    path.write_bytes(b"x")
    result = DocumentLoaderOperator().run({"path": str(path)})
    assert not result["documents"]
    assert "unsupported" in result["errors"][0]["error"]


def test_data_quality_deduplicates():
    result = DataQualityOperator().run(
        {"documents": [{"text": "同一 文本"}, {"text": "同一   文本"}, {"text": "不同"}]}
    )
    assert result["quality_report"]["duplicate_records"] == 1
    assert len(result["documents"]) == 2


def test_data_quality_empty_and_missing_fields():
    result = DataQualityOperator().run(
        {
            "documents": [{"text": ""}, {"text": "有效"}],
            "required_fields": ["fileName"],
        }
    )
    assert result["quality_report"]["empty_records"] == 1
    assert result["quality_report"]["missing_field_records"] == 1


@pytest.mark.parametrize(
    "dag",
    [
        [{"id": "a", "op": "x", "deps": ["missing"]}],
        [{"id": "a", "op": "x"}, {"id": "a", "op": "x"}],
        [{"id": "a", "op": "x", "deps": ["b"]}, {"id": "b", "op": "x", "deps": ["a"]}],
    ],
)
def test_topological_order_rejects_invalid_dags(dag):
    with pytest.raises(ValueError):
        topological_order(dag)


class _PassOperator(BaseOperator):
    def __init__(self, name="test_pass"):
        self.meta = OperatorMeta(
            name=name,
            input_schema={"type": "object"},
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            description="test",
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        return {"ok": True}


class _FailOperator(_PassOperator):
    def run(self, inputs: dict, **kwargs) -> dict:
        raise RuntimeError("planned failure")


class _RetryOperator(_PassOperator):
    def run(self, inputs: dict, **kwargs) -> dict:
        if not inputs.get("recover"):
            raise RuntimeError("needs adjusted args")
        return {"ok": True}


def test_dag_fallback_operator_recovery():
    register(_FailOperator("test_fail_fallback"))
    register(_PassOperator("test_fallback"))
    dag = [
        {
            "id": "a",
            "op": "test_fail_fallback",
            "deps": [],
            "on_error": {"fallback_op": "test_fallback"},
        }
    ]
    result = DAGExecutor(max_retries=0).run(dag, {}, verbose=False)
    assert result["states"]["a"]["status"] == NodeStatus.SUCCESS
    assert result["recoveries"][0]["strategy"] == "fallback_operator"


def test_dag_retry_with_adjusted_args():
    register(_RetryOperator("test_retry"))
    dag = [{"id": "a", "op": "test_retry", "deps": [], "retry_args": [{"recover": True}]}]
    result = DAGExecutor(max_retries=1).run(dag, {}, verbose=False)
    assert result["states"]["a"]["status"] == NodeStatus.SUCCESS
    assert result["states"]["a"]["attempts"] == 2


def test_dag_skips_failed_dependents():
    register(_FailOperator("test_fail_skip"))
    register(_PassOperator("test_downstream"))
    dag = [
        {"id": "a", "op": "test_fail_skip", "deps": []},
        {"id": "b", "op": "test_downstream", "deps": ["a"]},
    ]
    result = DAGExecutor(max_retries=0).run(dag, {}, verbose=False)
    assert result["states"]["a"]["status"] == NodeStatus.FAILED
    assert result["states"]["b"]["status"] == NodeStatus.SKIPPED


def test_graph_incremental_delta_and_audit():
    graph = LocalGraphStore()
    graph.begin_revision("test")
    graph.upsert_triple(
        "高血压", "Disease", "recommend_drug", "氨氯地平", "Drug",
        0.9, "guideline.txt", extractor="unit", chunk_id="c1",
    )
    delta = graph.commit_revision()
    audit = graph.audit()
    assert delta["added_entity_count"] == 2
    assert delta["added_triple_count"] == 1
    assert audit["provenance_coverage"] == 1.0
    assert audit["illegal_triples"] == 0


def test_graph_reasoning_paths():
    graph = LocalGraphStore()
    graph.upsert_triple("高血压", "Disease", "recommend_drug", "氨氯地平", "Drug", 0.9, "a")
    graph.upsert_triple("氨氯地平", "Drug", "contraindication", "低血压", "Disease", 0.8, "b")
    paths = graph.traverse_paths("高血压", hops=2)
    assert any(path["hops"] == 2 and "contraindication" in path["relations"] for path in paths)


def test_graph_export_has_metadata(tmp_path):
    graph = LocalGraphStore()
    graph.upsert_triple("高血压", "Disease", "has_symptom", "头痛", "Symptom", 1.0, "a")
    path = tmp_path / "graph.json"
    graph.export_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["schema_version"] == "2.0.0"


@pytest.mark.parametrize(
    ("triple", "expected"),
    [
        ({"relation": "has_symptom"}, "有症状"),  # CM3KG ontology label
        ({"relation": "cmeie:pathological_classification"}, "病理分型"),  # CMeIE key
        ({"relation": "has_symptom", "relation_zh": "自定义标签"}, "自定义标签"),  # explicit wins
        ({"relation": "unmapped_key"}, "unmapped_key"),  # graceful fallback to raw key
    ],
)
def test_qa_relation_label_covers_both_graph_flavours(triple, expected):
    """graph.json edges carry `relation_zh`; graph_scaled.json edges carry only a
    raw `cmeie:xxx` key. Without this fallback chain, QAAgent's prompt (and the
    LLM's composed answer) would show literal strings like
    "cmeie:pathological_classification" for anything sourced from the
    self-produced graph."""
    from medigraph.agents.qa_agent import _relation_label

    assert _relation_label(triple) == expected


def test_qa_refuses_low_confidence_evidence():
    graph = LocalGraphStore()
    graph.upsert_triple("高血压", "Disease", "has_symptom", "头痛", "Symptom", 0.2, "weak")
    result = QAAgent(llm=_NoLLM(), store=graph, min_answer_confidence=0.8).answer("高血压有哪些症状？")
    assert result["refused"]
    assert result["answer_confidence"]["grade"] == "low"


def _analytics_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT INTO patient_visits VALUES(1,1,60,'女','高血压','心内科','2024-01-01',100)"
    )
    connection.execute("INSERT INTO prescriptions VALUES(1,1,'二甲双胍',7)")
    connection.execute("INSERT INTO lab_tests VALUES(1,1,'血常规',1)")
    connection.execute("INSERT INTO kg_entities VALUES('高血压','Disease')")
    connection.commit()
    connection.close()


@pytest.mark.parametrize(
    ("question", "fragment"),
    [
        ("总共有多少次就诊", "COUNT(*)"),
        ("60岁以上的患者有多少人次", "age>60"),
        ("每个科室的就诊量是多少", "GROUP BY department"),
        ("2024年每个月的就诊量", "substr(visit_date"),
        ("开具次数最多的3种药物", "LIMIT 3"),
        ("每个检查项目的异常次数是多少", "SUM(abnormal)"),
    ],
)
def test_nl2sql_deterministic_templates(tmp_path, question, fragment):
    db = tmp_path / "analytics.db"
    _analytics_db(db)
    result = NL2SQL(str(db), llm=_NoLLM()).query(question)
    assert fragment in result["sql"]
    assert result["generation_mode"] == "deterministic_template"


def test_nl2sql_drug_count_preserves_roman_numeral_case(tmp_path):
    """Regression: the drug-count router path must not silently rewrite a
    matched drug name to a nonexistent one.

    `_deterministic_sql` lower-cases the question for keyword matching, and
    Python's str.lower() remaps Unicode Roman numerals (Ⅰ U+2160 -> ⅰ U+2170) --
    a suffix common in Chinese drug names ("硝苯地平缓释片Ⅰ" vs "...Ⅱ"). A regex
    fallback used to capture the drug name straight from that lower-cased text
    and overwrite the case-exact vocabulary match, producing a query for a drug
    string that cannot exist in the database. Found via the 128-question stress
    set after curating the disease vocabulary (see relational.DEFAULT_MAX_DISEASES).
    """
    db = tmp_path / "analytics.db"
    _analytics_db(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO prescriptions VALUES(2,1,'硝苯地平缓释片Ⅰ',14)"
        )
        connection.commit()

    result = NL2SQL(str(db), llm=_NoLLM()).query("硝苯地平缓释片Ⅰ一共被开具了多少次")
    assert "drug='硝苯地平缓释片Ⅰ'" in result["sql"], result["sql"]
    assert result["generation_mode"] == "deterministic_template"
    assert result["rows"] == [(1,)], "the case-correct literal must match the inserted row"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("科室疾病热力矩阵", "heatmap"),
        ("年龄与费用相关性散点图", "scatter"),
        ("每月趋势", "line"),
        ("性别比例", "pie"),
    ],
)
def test_extended_chart_picker(question, expected):
    rows = [("A", "B", 1)] if expected == "heatmap" else [(1, 2)]
    columns = ["x", "y", "value"] if expected == "heatmap" else ["x", "y"]
    assert pick_chart_type(question, columns, rows) == expected
