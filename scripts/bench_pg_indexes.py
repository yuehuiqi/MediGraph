r"""Measure what the PostgreSQL backend actually buys: indexes and pooling.

Two experiments, both with EXPLAIN ANALYZE / wall-clock evidence:

1. **Indexes** -- the analytics schema is rebuilt at a realistic scale
   (default 200k visits, ~280k prescriptions, ~300k lab rows), the NL2SQL
   router's real query shapes are EXPLAIN ANALYZEd without indexes, then the
   composite indexes from `pg_relational.INDEX_DDL` are created and the same
   queries re-measured. Per query the report keeps median execution time and the
   plan's node type, so "index helped" is visible as Seq Scan -> Index/Bitmap
   Scan rather than asserted.

   The 600-row demo DB is deliberately NOT what is measured: at that size a
   sequential scan beats an index and the honest result would be "indexes are
   noise". Index value is a function of table size; the report says so.

2. **Pooling** -- the same short query run N times over the shared
   `ConnectionPool` vs a fresh `psycopg.connect()` per query. The delta is the
   connection setup cost that pooling amortises.

The database is restored to the standard 600-row build (with indexes) at the end,
so demos and the service see the same data as SQLite afterwards.

    python scripts/bench_pg_indexes.py --visits 200000
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402

from config.settings import OUTPUTS_DIR, get_analytics_config  # noqa: E402
from medigraph.analysis import pg_relational as pg  # noqa: E402
from medigraph.graph.local_store import LocalGraphStore  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

#: The router's actual query shapes (see nl2sql._deterministic_sql / _FEWSHOT).
QUERIES: dict[str, str] = {
    "point_filter_age": (
        "SELECT AVG(cost) FROM patient_visits WHERE disease = '高血压' AND age > 60"
    ),
    "group_by_department": (
        "SELECT department, COUNT(*) AS cnt FROM patient_visits "
        "GROUP BY department ORDER BY cnt DESC"
    ),
    "month_trend": (
        "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visits "
        "FROM patient_visits GROUP BY month ORDER BY month"
    ),
    "drug_point_lookup": "SELECT COUNT(*) FROM prescriptions WHERE drug = '二甲双胍'",
    "abnormal_join": (
        "SELECT v.disease, COUNT(*) AS n FROM patient_visits v "
        "JOIN lab_tests t ON v.visit_id = t.visit_id "
        "WHERE t.abnormal = 1 GROUP BY v.disease ORDER BY n DESC LIMIT 5"
    ),
    "kg_head_relation": (
        "SELECT tail FROM kg_triples WHERE head = '高血压' AND relation = 'recommend_drug'"
    ),
}


def measure(sql: str, repeats: int) -> dict:
    runs = [pg.explain_analyze(sql) for _ in range(repeats)]
    times = sorted(run["execution_ms"] for run in runs)
    return {
        "execution_ms_p50": round(statistics.median(times), 3),
        "execution_ms_min": round(times[0], 3),
        "node_type": runs[-1]["node_type"],
        # First scan node below the root tells the story for aggregates.
        "scan": _first_scan(runs[-1]["plan"]),
    }


def _first_scan(plan: dict) -> str:
    node = plan.get("Node Type", "")
    if "Scan" in node:
        target = plan.get("Relation Name") or plan.get("Index Name") or ""
        return f"{node}({target})" if target else node
    for child in plan.get("Plans", []) or []:
        found = _first_scan(child)
        if found:
            return found
    return ""


def bench_pool(config, count: int) -> dict:
    """Pooled vs connect-per-query for a trivial statement."""
    sql = "SELECT COUNT(*) FROM patient_visits WHERE disease = '高血压'"

    pooled: list[float] = []
    pool = pg.get_pool(config)
    for _ in range(count):
        started = time.perf_counter()
        with pool.connection() as conn:
            conn.execute(sql).fetchone()
        pooled.append(time.perf_counter() - started)

    fresh: list[float] = []
    for _ in range(count):
        started = time.perf_counter()
        with psycopg.connect(config.pg_dsn, connect_timeout=10) as conn:
            conn.execute(sql).fetchone()
        fresh.append(time.perf_counter() - started)

    def stats(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "p50_ms": round(ordered[len(ordered) // 2] * 1000, 3),
            "p99_ms": round(ordered[int(0.99 * (len(ordered) - 1))] * 1000, 3),
            "mean_ms": round(statistics.fmean(values) * 1000, 3),
        }

    pooled_stats, fresh_stats = stats(pooled), stats(fresh)
    return {
        "queries": count,
        "pooled": pooled_stats,
        "connect_per_query": fresh_stats,
        "p50_speedup_x": round(fresh_stats["p50_ms"] / pooled_stats["p50_ms"], 1)
        if pooled_stats["p50_ms"]
        else None,
    }


def main() -> None:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--visits", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pool-queries", type=int, default=200)
    parser.add_argument("--graph", default=str(Path(OUTPUTS_DIR) / "graph.json"))
    args = parser.parse_args()

    config = get_analytics_config()
    if not pg.ping(config):
        raise SystemExit(
            f"PostgreSQL not reachable at {config.pg_host}:{config.pg_port} "
            "(see .env.example for the container command)"
        )

    store = LocalGraphStore.load_json(args.graph)

    print(f"[1/4] building at scale: {args.visits} visits, no indexes ...", flush=True)
    build = pg.build_pg_db(store, n_visits=args.visits, with_indexes=False, config=config)
    print(f"      loaded in {build['load_seconds']}s "
          f"({build['n_prescriptions']} rx, {build['n_lab_tests']} labs)")

    print(f"[2/4] EXPLAIN ANALYZE without indexes (p50 of {args.repeats}) ...", flush=True)
    before = {name: measure(sql, args.repeats) for name, sql in QUERIES.items()}

    print("[3/4] creating indexes, re-measuring ...", flush=True)
    started = time.perf_counter()
    pg.create_indexes(config)
    index_seconds = round(time.perf_counter() - started, 3)
    after = {name: measure(sql, args.repeats) for name, sql in QUERIES.items()}

    print(f"[4/4] pooled vs connect-per-query ({args.pool_queries} each) ...", flush=True)
    pool_bench = bench_pool(config, args.pool_queries)

    rows = {}
    for name in QUERIES:
        b, a = before[name], after[name]
        speedup = round(b["execution_ms_p50"] / a["execution_ms_p50"], 1) if a["execution_ms_p50"] else None
        rows[name] = {
            "before_ms": b["execution_ms_p50"],
            "after_ms": a["execution_ms_p50"],
            "speedup_x": speedup,
            "scan_before": b["scan"],
            "scan_after": a["scan"],
        }
        print(f"  {name:22} {b['execution_ms_p50']:9.2f} -> {a['execution_ms_p50']:8.2f} ms "
              f"({speedup}x)  {b['scan']} -> {a['scan']}")

    report = {
        "scale": {"visits": args.visits, **{k: build[k] for k in ("n_prescriptions", "n_lab_tests", "n_kg_triples")}},
        "load_seconds_copy": build["load_seconds"],
        "index_build_seconds": index_seconds,
        "repeats": args.repeats,
        "queries": {name: QUERIES[name] for name in QUERIES},
        "results": rows,
        "connection_pooling": pool_bench,
        "interpretation": (
            "Measured at 200k visits because index benefit is a function of table "
            "size: on the 600-row demo DB a sequential scan wins and the honest "
            "result would be 'indexes are noise'. Point lookups and selective "
            "filters gain the most (Seq Scan -> Index/Bitmap Scan); whole-table "
            "aggregates (GROUP BY over every row, substr() trends) keep scanning "
            "by design and gain little -- both outcomes are listed. Pooling "
            "amortises per-connection setup; its win shows up in p50 of short "
            "queries."
        ),
    }
    path = write_json(report, Path(OUTPUTS_DIR) / "bench_pg_indexes.json")
    pool = pool_bench
    print(f"\n  pooling: p50 {pool['connect_per_query']['p50_ms']}ms -> "
          f"{pool['pooled']['p50_ms']}ms ({pool['p50_speedup_x']}x)")
    print(f"  (saved -> {path})")

    print("\nrestoring standard 600-visit build (with indexes) ...")
    pg.build_pg_db(store, n_visits=600, with_indexes=True, config=config)
    print("done")


if __name__ == "__main__":
    main()
