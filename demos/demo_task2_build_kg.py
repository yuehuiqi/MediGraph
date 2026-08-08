"""Task 2 demo (part 1): build a knowledge graph from raw documents.

Usage:
  python demos/demo_task2_build_kg.py --input data/raw_demo --max-docs 3

Produces by default:
  outputs/raw_demo_graph.json   -- nodes + edges from raw documents
  outputs/raw_demo_graph.html    -- interactive visualization
  outputs/raw_demo_kg_build_stats.json

The CM3KG main graph at outputs/graph.json is kept intact unless you explicitly
pass --graph-name graph.json.
The graph is also persisted in the active backend (local by default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, RAW_DEMO_DIR  # noqa: E402
from medigraph.agents.kg_agent import KGGenAgent  # noqa: E402
from medigraph.graph.local_store import LocalGraphStore  # noqa: E402
from medigraph.graph.vector_store import LocalVectorStore  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import iter_documents, write_json  # noqa: E402

enable_utf8()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task2 KG build demo")
    parser.add_argument("--input", default=str(RAW_DEMO_DIR))
    parser.add_argument("--max-docs", type=int, default=3)
    parser.add_argument("--graph-name", default="raw_demo_graph.json")
    parser.add_argument(
        "--max-chunks", type=int, default=8,
        help="cap chunks processed per document (bounds runtime; 0 = no cap)",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="load existing graph.json + vectors.json first, then add to them "
             "(demonstrates incremental knowledge-graph evolution)",
    )
    args = parser.parse_args()

    docs = iter_documents(args.input)
    if not docs:
        print(f"No documents under {args.input}. Run data/prepare_local.py first.")
        sys.exit(1)
    docs = docs[: args.max_docs]
    print(f"Building KG from {len(docs)} document(s) ...")

    # Incremental mode: seed the agent with the previously built graph/vectors.
    preloaded_store = None
    preloaded_vectors = None
    graph_json = OUTPUTS_DIR / Path(args.graph_name).name
    if graph_json.suffix.lower() != ".json":
        graph_json = graph_json.with_suffix(".json")
    vectors_json = graph_json.with_name(f"{graph_json.stem}_vectors.json")
    if args.incremental and graph_json.exists():
        preloaded_store = LocalGraphStore.load_json(graph_json)
        print(f"[incremental] loaded existing graph: {preloaded_store.stats()['num_entities']} entities")
        if vectors_json.exists():
            preloaded_vectors = LocalVectorStore.load_json(vectors_json)

    agent = KGGenAgent(store=preloaded_store)
    if preloaded_vectors is not None and agent.vector_store is not None:
        agent.vector_store = preloaded_vectors
    stats = agent.build(docs, verbose=True, max_chunks_per_doc=args.max_chunks or None)

    print("\n========== BUILD STATS ==========")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    write_json(stats, OUTPUTS_DIR / f"{graph_json.stem}_build_stats.json")

    # Persist the vector index for hybrid GraphRAG in the QA demo.
    if agent.vector_store is not None and agent.vector_store.size:
        vp = agent.vector_store.save_json(OUTPUTS_DIR / "vectors.json")
        print(f"Vector index -> {vp} ({agent.vector_store.size} chunks)")

    # Export graph artifacts (local backend exposes export helpers).
    store = agent.store
    if isinstance(store, LocalGraphStore):
        gj = store.export_json(graph_json)
        print(f"\nGraph JSON  -> {gj}")
        try:
            gh = store.export_html(graph_json.with_suffix(".html"))
            print(f"Graph HTML  -> {gh}  (open in a browser)")
        except Exception as exc:  # noqa: BLE001
            print(f"(HTML export skipped: {exc})")
    else:
        print("\n(Neo4j backend active: inspect the graph in Neo4j Browser at http://localhost:7474)")


if __name__ == "__main__":
    main()
