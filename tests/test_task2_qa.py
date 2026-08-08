"""Offline tests for Task 2 intent-aware, traceable GraphRAG."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medigraph.agents.qa_agent import QAAgent, infer_intent_relations, rank_evidence, select_evidence
from medigraph.graph.local_store import LocalGraphStore


class _FakeStats:
    def summary(self) -> dict:
        return {"calls": 0}


class _FakeLLM:
    def __init__(self):
        self.stats = _FakeStats()

    def chat_json(self, prompt: str, **kwargs) -> dict:
        return {
            "entities": [
                {"name": "高血压", "type": "Disease", "confidence": 0.99},
            ]
        }

    def chat(self, prompt: str, **kwargs) -> str:
        return "高血压可使用氨氯地平。依据：高血压--[推荐药物]-->氨氯地平。"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []


def _graph() -> LocalGraphStore:
    graph = LocalGraphStore()
    graph.upsert_triple("高血压", "Disease", "has_symptom", "头痛", "Symptom", 0.95, "demo.txt")
    graph.upsert_triple("高血压", "Disease", "recommend_drug", "氨氯地平", "Drug", 0.91, "demo.txt")
    graph.upsert_triple("高血压", "Disease", "need_examination", "动态血压", "Examination", 0.90, "demo.txt")
    return graph


def test_intent_relation_detection():
    """Each rule now pairs the CM3KG-import key with its CMeIE-V2 counterpart
    (see _INTENT_RULES's docstring): graph.json and graph_scaled.json use
    different relation vocabularies for the same concept, so a question routed
    against either graph must resolve intent either way."""
    relations = infer_intent_relations("高血压有哪些症状和推荐药物？")
    assert relations == [
        "has_symptom", "cmeie:clinical_manifestation",
        "recommend_drug", "cmeie:drug_treatment",
    ]


def test_relation_aware_evidence_ranking():
    evidence = _graph().neighbors("高血压")
    ranked = rank_evidence(evidence, ["recommend_drug"])
    assert ranked[0]["relation"] == "recommend_drug"


def test_qa_returns_traceable_citations():
    result = QAAgent(llm=_FakeLLM(), store=_graph(), hops=1).answer("高血压推荐什么药物？")
    assert result["retrieval_mode"] == "graph"
    assert result["intent_relations"] == ["recommend_drug", "cmeie:drug_treatment"]
    assert result["citations"][0]["relation"] == "recommend_drug"
    assert result["citations"][0]["source"] == "demo.txt"


def test_evidence_balances_multiple_requested_relations():
    evidence = []
    for index in range(20):
        evidence.append({"relation": "has_symptom", "confidence": 1.0, "head": "D", "tail": f"S{index}"})
    for index in range(20):
        evidence.append({"relation": "recommend_drug", "confidence": 1.0, "head": "D", "tail": f"M{index}"})
    selected = select_evidence(evidence, ["has_symptom", "recommend_drug"])
    assert len([item for item in selected if item["relation"] == "has_symptom"]) == 8
    assert len([item for item in selected if item["relation"] == "recommend_drug"]) == 8


def test_direct_outgoing_evidence_wins_over_two_hop_neighbours():
    evidence = [
        {"relation": "complication", "confidence": 1.0, "head": "其他疾病", "tail": "肾衰竭"},
        {"relation": "complication", "confidence": 1.0, "head": "高血压", "tail": "中风", "source": "CM3KG"},
        {"relation": "complication", "confidence": 1.0, "head": "中风", "tail": "昏迷"},
    ]

    ranked = rank_evidence(evidence, ["complication"], ["高血压"])
    selected = select_evidence(ranked, ["complication"], anchors=["高血压"])

    assert selected == [evidence[1]]
