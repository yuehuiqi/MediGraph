"""Offline tests for graph analytics: PageRank, degree centrality, community
detection, shortest path -- and the AnalysisAgent routing that reaches them.
"""
from __future__ import annotations

import pytest

from medigraph.analysis.graph_algorithms import (
    degree_centrality_ranking,
    detect_communities,
    pagerank_ranking,
    shortest_path,
)
from medigraph.graph.local_store import LocalGraphStore


@pytest.fixture
def hub_graph():
    """A small graph with an obvious hub (糖尿病, degree 6) and two loosely
    connected satellite clusters, so ranking/community results are checkable
    against a known-correct answer instead of just "did it run"."""
    store = LocalGraphStore()
    hub_edges = [
        ("糖尿病", "Disease", "has_symptom", "多饮", "Symptom"),
        ("糖尿病", "Disease", "has_symptom", "多尿", "Symptom"),
        ("糖尿病", "Disease", "has_symptom", "体重下降", "Symptom"),
        ("糖尿病", "Disease", "recommend_drug", "二甲双胍", "Drug"),
        ("糖尿病", "Disease", "recommend_drug", "胰岛素", "Drug"),
        ("糖尿病", "Disease", "complication", "糖尿病肾病", "Disease"),
    ]
    for head, head_type, relation, tail, tail_type in hub_edges:
        store.upsert_triple(head, head_type, relation, tail, tail_type, confidence=1.0)
    # A second, disconnected cluster so community detection has >1 community.
    store.upsert_triple("骨折", "Disease", "has_symptom", "肿胀", "Symptom", confidence=1.0)
    store.upsert_triple("骨折", "Disease", "recommend_drug", "布洛芬", "Drug", confidence=1.0)
    store.upsert_triple("骨折", "Disease", "has_symptom", "疼痛", "Symptom", confidence=1.0)
    return store


def test_pagerank_rewards_a_shared_in_edge_target(hub_graph):
    """PageRank accumulates rank at nodes that are *pointed to*; this project's
    edges point disease -> attribute, so a symptom shared by two diseases (two
    in-edges) must outrank a symptom only one disease points to (one in-edge) --
    the directed-graph analogue of "the hub disease ranks first" that a naive
    hub-node assumption gets backwards on this edge direction (see
    graph_algorithms.pagerank_ranking's docstring)."""
    hub_graph.upsert_triple("骨折", "Disease", "has_symptom", "多饮", "Symptom", confidence=1.0)
    ranked = {item["name"]: item["pagerank"] for item in pagerank_ranking(hub_graph, top_n=20)}
    assert ranked["多饮"] > ranked["多尿"]  # 多饮 now has 2 in-edges, 多尿 has 1


def test_pagerank_disease_filter_reflects_in_edges_not_out_edges(hub_graph):
    """Filtered to Disease, 糖尿病肾病 (an in-edge target via `complication`) must
    outrank 骨折 (a pure source node, no in-edges at all)."""
    ranked = {item["name"]: item["pagerank"] for item in pagerank_ranking(hub_graph, top_n=20, entity_type="Disease")}
    assert ranked["糖尿病肾病"] > ranked["骨折"]


def test_pagerank_entity_type_filter(hub_graph):
    ranked = pagerank_ranking(hub_graph, top_n=10, entity_type="Drug")
    assert ranked
    assert all(item["type"] == "Drug" for item in ranked)


def test_degree_centrality_ranks_the_hub_first(hub_graph):
    ranked = degree_centrality_ranking(hub_graph, top_n=10)
    assert ranked[0]["name"] == "糖尿病"
    assert ranked[0]["degree"] == 6


def test_community_detection_separates_disconnected_clusters(hub_graph):
    communities = detect_communities(hub_graph, top_n=5, min_size=2)
    assert len(communities) >= 2
    all_members = {name for community in communities for name in community["representative_entities"]}
    assert "糖尿病" in all_members
    assert "骨折" in all_members
    # The hub cluster (6 edges) must be reported larger than the satellite (3).
    sizes = sorted(c["size"] for c in communities)
    assert sizes[-1] > sizes[0]


def test_community_min_size_drops_singletons(hub_graph):
    hub_graph.upsert_triple("孤立疾病", "Disease", "has_symptom", "孤立症状", "Symptom", confidence=1.0)
    communities = detect_communities(hub_graph, top_n=10, min_size=3)
    all_members = {name for community in communities for name in community["representative_entities"]}
    assert "孤立疾病" not in all_members


def test_shortest_path_direct_edge(hub_graph):
    result = shortest_path(hub_graph, "糖尿病", "多饮")
    assert result is not None
    assert result["hops"] == 1
    assert result["steps"][0]["relation"] == "has_symptom"


def test_shortest_path_multi_hop(hub_graph):
    # 二甲双胍 -[recommend_drug, reversed]- 糖尿病 -[complication]- 糖尿病肾病
    result = shortest_path(hub_graph, "二甲双胍", "糖尿病肾病")
    assert result is not None
    assert result["hops"] == 2
    assert result["path"][1] == "糖尿病"


def test_shortest_path_no_path_between_disconnected_clusters(hub_graph):
    assert shortest_path(hub_graph, "糖尿病", "骨折") is None


def test_shortest_path_unknown_entity(hub_graph):
    assert shortest_path(hub_graph, "糖尿病", "不存在的实体") is None


# --------------------------------------------------------------------------- #
# AnalysisAgent routing: algorithm-intent questions must reach GRAPH_ALGO, not
# fall into the 1-hop neighbor lookup (GRAPH) or NL2SQL (SQL).
# --------------------------------------------------------------------------- #
class _NoLLM:
    class _Stats:
        def summary(self) -> dict:
            return {}

    stats = _Stats()

    def chat_json(self, *args, **kwargs):
        return kwargs.get("default", {})

    def chat(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("routing test unexpectedly reached the LLM")


@pytest.mark.parametrize(
    ("question", "expected_algorithm"),
    [
        ("图谱中最核心的疾病是哪些？", "degree_centrality"),
        ("哪些疾病的PageRank最高？", "pagerank"),
        ("图谱中有哪些社区聚类？", "community"),
        ("高血压和糖尿病肾病之间的最短路径是什么？", "shortest_path"),
    ],
)
def test_plan_route_detects_graph_algorithm_intent(question, expected_algorithm):
    from medigraph.analysis.analysis_agent import AnalysisAgent

    agent = AnalysisAgent.__new__(AnalysisAgent)  # skip __init__'s DB/graph loading
    plan = AnalysisAgent.plan_route(agent, question)
    assert plan["route"] == "GRAPH_ALGO"


def test_graph_algorithm_answer_centrality_end_to_end(hub_graph):
    from medigraph.analysis.analysis_agent import AnalysisAgent

    agent = AnalysisAgent.__new__(AnalysisAgent)
    agent.store = hub_graph
    agent.llm = _NoLLM()
    result = AnalysisAgent._graph_algorithm_answer(agent, "图谱中最核心的疾病是哪些？")
    assert result["source"] == "GRAPH_ALGO"
    assert result["algorithm"] == "degree_centrality"
    assert result["rows"]
    assert result["rows"][0][0] == "糖尿病"


def test_graph_algorithm_answer_community_end_to_end(hub_graph):
    from medigraph.analysis.analysis_agent import AnalysisAgent

    agent = AnalysisAgent.__new__(AnalysisAgent)
    agent.store = hub_graph
    agent.llm = _NoLLM()
    result = AnalysisAgent._graph_algorithm_answer(agent, "图谱中有哪些社区聚类？")
    assert result["algorithm"] == "community"
    assert result["columns"] == ["community_id", "size", "representative_entities"]
