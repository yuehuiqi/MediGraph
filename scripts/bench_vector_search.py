r"""Measure the ANN trade-off: pgvector HNSW vs numpy brute force.

Protocol
--------
* N synthetic unit vectors (default 100k, dim 1024 -- the Qwen3-Embedding-0.6B
  width), deterministic seed.
* Ground truth = numpy exact top-k (this is also the `LocalVectorStore` code
  path, so the baseline latency IS the current production path's latency).
* pgvector HNSW queried at several `ef_search` values; per point:
  recall@10 against the exact top-10, and client-side latency p50/p99.

Recall\@10 here means |approx top-10 ∩ exact top-10| / 10 averaged over queries.
The point of publishing the whole curve instead of one number: ef_search is a
dial, and the "right" setting is whichever recall the application needs at the
latency it can afford.

    python scripts/bench_vector_search.py --n 100000 --queries 100
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, get_analytics_config  # noqa: E402
from medigraph.analysis.pg_relational import ping  # noqa: E402
from medigraph.graph.pg_vector_store import PgVectorStore  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

TABLE = "vector_bench"


def pct(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def main() -> None:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-values", type=int, nargs="+", default=[10, 20, 40, 100, 200])
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--distribution",
        choices=("clustered", "gaussian"),
        default="clustered",
        help=(
            "clustered (default): points around shared topic centres, matching the "
            "structure of real text embeddings (same-topic cosine ~0.65, cross-topic "
            "~0). gaussian: i.i.d. vectors -- ANN's adversarial worst case, where in "
            "high dimension all pairwise cosines concentrate near 0 and 'nearest' "
            "neighbours are barely nearer than everything else; kept as the floor."
        ),
    )
    parser.add_argument("--n-centers", type=int, default=1000)
    parser.add_argument("--cluster-sigma", type=float, default=0.7)
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help=(
            "Reuse vectors already COPYed into the bench table (they are "
            "regenerated deterministically from --seed for the ground truth, so "
            "the table content is identical)."
        ),
    )
    args = parser.parse_args()

    config = get_analytics_config()
    if not ping(config):
        raise SystemExit("PostgreSQL not reachable (see .env.example for the container command)")

    rng = np.random.default_rng(args.seed)
    print(
        f"[1/5] generating {args.n} x {args.dim} unit vectors ({args.distribution}) ...",
        flush=True,
    )

    def normalize(matrix: np.ndarray) -> np.ndarray:
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

    if args.distribution == "gaussian":
        corpus = normalize(rng.standard_normal((args.n, args.dim)).astype(np.float32))
        queries = normalize(rng.standard_normal((args.queries, args.dim)).astype(np.float32))
    else:
        # Topic-centre mixture: point = normalize(centre + sigma * g / sqrt(d)).
        # With sigma=0.7 a point's cosine to its own centre is ~1/sqrt(1+sigma^2)
        # ~0.82 and to same-cluster peers ~1/(1+sigma^2) ~0.67, while cross-cluster
        # cosines stay near 0 -- the shape real chunk embeddings have. Queries are
        # drawn by the same recipe (a question embeds near its topic's chunks).
        centres = normalize(rng.standard_normal((args.n_centers, args.dim)).astype(np.float32))
        assign = rng.integers(0, args.n_centers, size=args.n)
        noise = rng.standard_normal((args.n, args.dim)).astype(np.float32) / np.sqrt(args.dim)
        corpus = normalize(centres[assign] + args.cluster_sigma * noise)
        query_assign = rng.integers(0, args.n_centers, size=args.queries)
        query_noise = (
            rng.standard_normal((args.queries, args.dim)).astype(np.float32) / np.sqrt(args.dim)
        )
        queries = normalize(centres[query_assign] + args.cluster_sigma * query_noise)

    # ---- exact baseline: the LocalVectorStore code path (matrix @ q) -------- #
    print("[2/5] exact ground truth + brute-force latency ...", flush=True)
    truth: list[set[int]] = []
    brute_latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        scores = corpus @ query
        top = np.argpartition(-scores, args.k)[: args.k]
        top = top[np.argsort(-scores[top])]
        brute_latencies.append(time.perf_counter() - started)
        truth.append(set(int(i) for i in top))

    # ---- load + index ------------------------------------------------------- #
    store = PgVectorStore(
        dim=args.dim, config=config, m=args.m, ef_construction=args.ef_construction, table=TABLE
    )
    if args.skip_load and store.size == args.n:
        print(f"[3/5] reusing {store.size} loaded vectors (--skip-load)", flush=True)
        load_seconds = None
    else:
        print("[3/5] COPY into pgvector ...", flush=True)
        store.clear()
        started = time.perf_counter()
        batch = 10_000
        for offset in range(0, args.n, batch):
            chunk = corpus[offset : offset + batch]
            store.add(
                [str(i) for i in range(offset, offset + len(chunk))],
                ["bench"] * len(chunk),
                chunk.tolist(),
            )
        load_seconds = round(time.perf_counter() - started, 2)
        print(f"      loaded in {load_seconds}s")

    print("[4/5] building HNSW index ...", flush=True)
    started = time.perf_counter()
    store.create_index()
    index_seconds = round(time.perf_counter() - started, 2)
    print(f"      built in {index_seconds}s (m={args.m}, ef_construction={args.ef_construction})")

    # ---- sweep -------------------------------------------------------------- #
    print("[5/5] ef_search sweep ...", flush=True)
    curve = []
    for ef in args.ef_values:
        latencies: list[float] = []
        recalls: list[float] = []
        for index, query in enumerate(queries):
            started = time.perf_counter()
            hits = store.search(query.tolist(), k=args.k, ef_search=ef)
            latencies.append(time.perf_counter() - started)
            found = {int(hit["text"]) for hit in hits}
            recalls.append(len(found & truth[index]) / args.k)
        point = {
            "ef_search": ef,
            "recall_at_10": round(statistics.fmean(recalls), 4),
            "latency_ms_p50": round(pct(latencies, 0.50) * 1000, 2),
            "latency_ms_p99": round(pct(latencies, 0.99) * 1000, 2),
        }
        curve.append(point)
        print(f"  ef={ef:4d}  recall@10={point['recall_at_10']:.4f}  "
              f"p50={point['latency_ms_p50']:7.2f}ms  p99={point['latency_ms_p99']:7.2f}ms")

    brute = {
        "recall_at_10": 1.0,
        "latency_ms_p50": round(pct(brute_latencies, 0.50) * 1000, 2),
        "latency_ms_p99": round(pct(brute_latencies, 0.99) * 1000, 2),
    }
    print(f"  numpy exact       recall@10=1.0000  p50={brute['latency_ms_p50']:7.2f}ms  "
          f"p99={brute['latency_ms_p99']:7.2f}ms")

    report = {
        "n_vectors": args.n,
        "dim": args.dim,
        "n_queries": args.queries,
        "k": args.k,
        "distribution": args.distribution,
        "cluster_params": (
            {"n_centers": args.n_centers, "sigma": args.cluster_sigma}
            if args.distribution == "clustered"
            else None
        ),
        "hnsw": {"m": args.m, "ef_construction": args.ef_construction},
        "load_seconds_copy": load_seconds,
        "index_build_seconds": index_seconds,
        "numpy_bruteforce": brute,
        "hnsw_curve": curve,
        "interpretation": (
            "The numpy row is the LocalVectorStore production path: exact (recall "
            "1.0) but O(N) per query, holding the full matrix in the API process. "
            "The HNSW curve shows the ef_search dial: raise it for recall, lower it "
            "for latency. Latencies are client-side and include one pooled "
            "round-trip for pgvector. The 'clustered' distribution mimics real text "
            "embeddings (same-topic cosine ~0.67, cross-topic ~0); the 'gaussian' "
            "run (bench_vector_search_gaussian.json) is ANN's adversarial worst "
            "case -- i.i.d. high-dim vectors where all cosines concentrate near 0 "
            "and HNSW recall collapses -- kept published as the floor. Honest "
            "flip side: at 100k vectors a single in-process matmul is still "
            "latency-competitive with a pgvector round-trip; the index's wins are "
            "recall-at-scale headroom (numpy is O(N), HNSW ~O(log N)), keeping "
            "hundreds of MB of vectors out of the API process, and server-side "
            "concurrency."
        ),
    }
    path = write_json(report, Path(OUTPUTS_DIR) / f"bench_vector_search_{args.distribution}.json"
                      if args.distribution == "gaussian"
                      else Path(OUTPUTS_DIR) / "bench_vector_search.json")
    print(f"  (saved -> {path})")

    store.clear()
    print("bench table cleared")


if __name__ == "__main__":
    main()
