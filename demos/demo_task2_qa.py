"""Task 2 demo (part 2): GraphRAG question answering over the built graph.

Usage:
  python demos/demo_task2_qa.py --question "高血压有哪些症状和推荐药物？"
  python demos/demo_task2_qa.py            # runs a few sample questions

For the local backend it loads outputs/graph.json (produced by the build demo).
For the neo4j backend it queries the live database.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, get_graph_config  # noqa: E402
from medigraph.agents.qa_agent import QAAgent  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

enable_utf8()

_SAMPLE_QUESTIONS = [
    "高血压有哪些症状和推荐药物？",
    "糖尿病需要做哪些检查？",
    "高血压会有哪些并发症？",
]


def _build_store():
    cfg = get_graph_config()
    if cfg.backend == "neo4j":
        from medigraph.graph.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
    from medigraph.graph.local_store import LocalGraphStore

    graph_json = OUTPUTS_DIR / "graph.json"
    if not graph_json.exists():
        print(f"Local graph not found at {graph_json}. Run demo_task2_build_kg.py first.")
        sys.exit(1)
    return LocalGraphStore.load_json(graph_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task2 GraphRAG QA demo")
    parser.add_argument("--question", default=None)
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()

    store = _build_store()

    # Load the vector index if present -> hybrid GraphRAG (graph + vector).
    vector_store = None
    vec_path = OUTPUTS_DIR / "vectors.json"
    if vec_path.exists():
        from medigraph.graph.vector_store import LocalVectorStore
        vector_store = LocalVectorStore.load_json(vec_path)
        print(f"Loaded vector index: {vector_store.size} chunks (hybrid retrieval ON)")

    agent = QAAgent(store=store, hops=args.hops, vector_store=vector_store)

    questions = [args.question] if args.question else _SAMPLE_QUESTIONS
    for q in questions:
        print(f"\n================ Q: {q} ================")
        res = agent.answer(q, verbose=True)
        print(f"\nAnswer:\n{res['answer']}")
        print(f"\nResolved entities: {res['resolved_entities']}")
        print(f"Evidence triples used: {len(res['evidence'])}")
        if res["evidence"]:
            print("Sample evidence:")
            for t in res["evidence"][:8]:
                print(f"  - {t['head']} --[{t.get('relation_zh', t['relation'])}]--> {t['tail']}  (来源={t.get('source','')})")


if __name__ == "__main__":
    main()
