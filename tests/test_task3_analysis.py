"""Offline tests for Task 3 routing, provenance and safe visualization."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medigraph.analysis.analysis_agent import AnalysisAgent
from medigraph.analysis.nl2sql import NL2SQL
from medigraph.analysis.viz import render_report
from medigraph.graph.local_store import LocalGraphStore


class _Stats:
    def summary(self) -> dict:
        return {"calls": 0}


class _FakeLLM:
    def __init__(self):
        self.stats = _Stats()

    def chat_json(self, prompt: str, **kwargs) -> dict:
        return {"route": "SQL", "reason": "fallback"}

    def chat(self, prompt: str, **kwargs) -> str:
        return "测试洞察"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []


def _db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE patient_visits(visit_id INTEGER, department TEXT)")
    conn.execute("INSERT INTO patient_visits VALUES(1,'心内科')")
    conn.execute(
        "CREATE TABLE kg_triples("
        "head TEXT, head_type TEXT, relation TEXT, tail TEXT, tail_type TEXT, "
        "confidence REAL, source TEXT)"
    )
    conn.executemany(
        "INSERT INTO kg_triples VALUES(?,?,?,?,?,?,?)",
        [
            ("高血压", "Disease", "has_symptom", "头痛", "Symptom", 1.0, "test"),
            ("高血压", "Disease", "has_symptom", "头晕", "Symptom", 1.0, "test"),
            ("糖尿病", "Disease", "has_symptom", "多饮", "Symptom", 1.0, "test"),
        ],
    )
    conn.commit()
    conn.close()


def test_deterministic_task3_route(tmp_path):
    db = tmp_path / "a.db"
    _db(db)
    agent = AnalysisAgent(str(db), llm=_FakeLLM())
    assert agent.plan_route("各疾病的就诊人次是多少")["route"] == "SQL"
    assert agent.plan_route("高血压有哪些并发症和推荐用药")["route"] == "GRAPH"
    assert agent.plan_route("统计知识图谱里关联症状最多的 Top 10 疾病")["route"] == "SQL"


def test_kg_symptom_top10_uses_deterministic_sql(tmp_path):
    db = tmp_path / "a.db"
    _db(db)
    result = NL2SQL(str(db), llm=_FakeLLM()).query(
        "统计知识图谱里关联症状最多的 Top 10 疾病，用柱状图展示"
    )
    assert result["generation_mode"] == "deterministic_template"
    assert "FROM kg_triples" in result["sql"]
    assert "relation = 'has_symptom'" in result["sql"]
    assert result["rows"][0] == ("高血压", 2)


def test_generic_disease_symptom_network_has_graph_rows(tmp_path):
    db = tmp_path / "a.db"
    _db(db)
    graph = LocalGraphStore()
    graph.upsert_triple("高血压", "Disease", "has_symptom", "头痛", "Symptom", 1.0, "test")
    graph.upsert_triple("高血压", "Disease", "has_symptom", "头晕", "Symptom", 1.0, "test")
    graph.upsert_triple("糖尿病", "Disease", "has_symptom", "多饮", "Symptom", 1.0, "test")
    graph_path = tmp_path / "graph.json"
    graph.export_json(graph_path)
    agent = AnalysisAgent(str(db), graph_json=str(graph_path), llm=_FakeLLM())
    result = agent.analyze("把疾病—症状的关联关系画成一张关系图")
    assert result["route"] == "GRAPH"
    assert result["chart_type"] == "graph"
    assert len(result["rows"]) == 3


def test_anchored_complication_query_keeps_outgoing_direction(tmp_path):
    db = tmp_path / "a.db"
    _db(db)
    graph = LocalGraphStore()
    graph.upsert_triple("高血压", "Disease", "complication", "冠心病", "Disease", 0.92, "hypertension.txt")
    graph.upsert_triple("慢性肾炎", "Disease", "complication", "高血压", "Disease", 0.88, "renal.txt")
    graph.upsert_triple("高血压", "Disease", "complication", "高血压", "Disease", 0.51, "noise.txt")
    graph.upsert_triple("高血压", "Disease", "has_symptom", "头晕", "Symptom", 0.95, "hypertension.txt")
    graph_path = tmp_path / "graph.json"
    graph.export_json(graph_path)

    agent = AnalysisAgent(str(db), graph_json=str(graph_path), llm=_FakeLLM())
    result = agent.analyze("高血压有哪些并发症？")

    assert result["route"] == "GRAPH"
    assert result["rows"] == [("高血压", "并发", "冠心病")]
    assert result["citations"][0]["source"] == "hypertension.txt"


def test_top5_patient_cost_uses_deterministic_sql(tmp_path):
    db = tmp_path / "analytics.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE patient_visits("
        "visit_id INTEGER, patient_id INTEGER, age INTEGER, gender TEXT, "
        "disease TEXT, department TEXT, visit_date TEXT, cost REAL)"
    )
    connection.executemany(
        "INSERT INTO patient_visits VALUES(?,?,?,?,?,?,?,?)",
        [
            (1, 1001, 60, "女", "高血压", "心内科", "2024-01-01", 100.0),
            (2, 1002, 62, "男", "糖尿病", "内分泌科", "2024-01-02", 300.0),
        ],
    )
    connection.commit()
    connection.close()
    result = NL2SQL(str(db), llm=_FakeLLM()).query("住院费用最高的前5名患者")
    assert result["generation_mode"] == "deterministic_template"
    assert result["sql"] == (
        "SELECT patient_id, cost FROM patient_visits ORDER BY cost DESC LIMIT 5"
    )
    assert result["rows"][0] == (1002, 300.0)


def test_task3_report_escapes_user_html(tmp_path):
    out = tmp_path / "report.html"
    render_report(out, "<script>alert(1)</script>", "bar", ["x", "y"], [("<b>A</b>", 1)], "<img src=x>")
    content = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
    assert "&lt;b&gt;A&lt;/b&gt;" in content
