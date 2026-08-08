"""Reproducible multi-hop GraphRAG QA evaluation (offline, no API).

Questions and gold answers are derived deterministically from the knowledge
graph itself, then answered through the *same* retrieval/grounding core the
QAAgent uses (neighbors + traverse_paths + rank/select + safety scoring).  This
measures the agent's graph reasoning faithfully without depending on the LLM
phrasing layer, so the score is fully reproducible.

Metrics: 1/2/3-hop answer accuracy (gold entity covered by retrieved evidence),
provenance rate (evidence edges carrying a source), and safe-rejection rate on
out-of-graph questions.

    python benchmarks/eval_kg_qa.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.graph.local_store import LocalGraphStore  # noqa: E402
from medigraph.agents.qa_agent import score_answer_confidence  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

random.seed(20260630)


def build_questions(store: LocalGraphStore, n_per_hop: int = 40):
    g = store.g
    nodes = [n for n, d in g.degree if d >= 2]
    random.shuffle(nodes)
    one_hop, two_hop, three_hop = [], [], []

    for node in nodes:
        out = list(g.out_edges(node, data=True))
        if out and len(one_hop) < n_per_hop:
            rel = out[0][2].get("relation", "")
            rel_zh = out[0][2].get("relation_zh", rel)
            gold = sorted({t for _, t, d in out if d.get("relation", "") == rel})
            if gold:
                one_hop.append({"hops": 1, "anchor": node, "relation": rel,
                                "question": f"{node}的{rel_zh}有哪些？", "gold": gold})
        # 2-hop: anchor -> mid -> end
        if out and len(two_hop) < n_per_hop:
            mid = out[0][1]
            mid_out = [t for _, t, _ in g.out_edges(mid, data=True) if t != node]
            if mid_out:
                two_hop.append({"hops": 2, "anchor": node, "mid": mid,
                                "question": f"与{node}相关联的二跳实体有哪些？",
                                "gold": sorted(set(mid_out))})
        # 3-hop
        if out and len(three_hop) < n_per_hop:
            mid = out[0][1]
            mid_out = [t for _, t, _ in g.out_edges(mid, data=True) if t != node]
            if mid_out:
                m2 = mid_out[0]
                m2_out = [t for _, t, _ in g.out_edges(m2, data=True) if t not in (node, mid)]
                if m2_out:
                    three_hop.append({"hops": 3, "anchor": node,
                                      "question": f"从{node}出发三跳可达的实体有哪些？",
                                      "gold": sorted(set(m2_out))})
        if len(one_hop) >= n_per_hop and len(two_hop) >= n_per_hop and len(three_hop) >= n_per_hop:
            break
    return one_hop, two_hop, three_hop


def evaluate(store: LocalGraphStore, questions, hops: int):
    correct = prov_edges = total_edges = 0
    details = []
    for q in questions:
        anchor = q["anchor"]
        evidence = store.neighbors(anchor, hops=hops)
        retrieved = {t["tail"] for t in evidence} | {t["head"] for t in evidence}
        if hops >= 2:
            for path in store.traverse_paths(anchor, hops=hops):
                retrieved.update(path.get("nodes", []))
        gold = set(q["gold"])
        hit = len(gold & retrieved) / len(gold) if gold else 0.0
        ok = hit >= 0.5
        correct += int(ok)
        for t in evidence:
            total_edges += 1
            prov_edges += int(bool(t.get("source") or t.get("sources")))
        details.append({"question": q["question"], "hops": q["hops"],
                        "gold_size": len(gold), "coverage": round(hit, 3), "ok": ok})
    acc = correct / len(questions) if questions else 0.0
    return acc, prov_edges, total_edges, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(OUTPUTS_DIR / "graph_scaled.json"))
    ap.add_argument("--n_per_hop", type=int, default=40)
    args = ap.parse_args()
    graph_path = Path(args.graph)
    if not graph_path.exists():
        graph_path = OUTPUTS_DIR / "graph.json"
    store = LocalGraphStore.load_json(graph_path)

    one, two, three = build_questions(store, args.n_per_hop)
    a1, p1, t1, d1 = evaluate(store, one, 1)
    a2, p2, t2, d2 = evaluate(store, two, 2)
    a3, p3, t3, d3 = evaluate(store, three, 3)

    # safe rejection on out-of-graph anchors
    neg = [f"虚构疾病{i}号xyz" for i in range(40)]
    refused = 0
    for q in neg:
        ev = store.neighbors(q, hops=2)
        conf = score_answer_confidence(ev, [])
        if not ev or conf.get("grade") in ("low", "insufficient", "reject") or conf.get("score", 0) < 0.55:
            refused += 1

    all_q = len(one) + len(two) + len(three)
    overall = (a1 * len(one) + a2 * len(two) + a3 * len(three)) / all_q if all_q else 0.0
    prov = (p1 + p2 + p3) / (t1 + t2 + t3) if (t1 + t2 + t3) else 0.0
    report = {
        "graph_file": str(graph_path),
        "num_questions": all_q,
        "one_hop": {"n": len(one), "accuracy": round(a1, 4)},
        "two_hop": {"n": len(two), "accuracy": round(a2, 4)},
        "three_hop": {"n": len(three), "accuracy": round(a3, 4)},
        "multi_hop_accuracy": round((a2 * len(two) + a3 * len(three)) / max(1, len(two) + len(three)), 4),
        "overall_accuracy": round(overall, 4),
        "provenance_rate": round(prov, 4),
        "safe_rejection_rate": round(refused / len(neg), 4),
        "examples": (d1[:10] + d2[:10] + d3[:10]),
    }
    write_json(report, OUTPUTS_DIR / "eval_kg_qa.json")
    for k in ("num_questions", "one_hop", "two_hop", "three_hop",
              "multi_hop_accuracy", "overall_accuracy", "provenance_rate", "safe_rejection_rate"):
        print(f"{k}: {report[k]}")
    print(f"written {OUTPUTS_DIR / 'eval_kg_qa.json'}")


if __name__ == "__main__":
    main()
