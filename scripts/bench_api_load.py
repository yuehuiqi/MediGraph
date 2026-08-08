r"""Concurrent load test for the HTTP service.

Targets the deterministic NL2SQL route by default: it exercises the full request
path (middleware, metrics, validation, schema linking, guarded SQL execution)
without spending LLM quota, so the numbers are reproducible and the run is free.

    python service/main.py                                  # in another shell
    python scripts/bench_api_load.py --concurrency 16 --requests 400

Reports throughput and the latency distribution. Percentiles are computed over
per-request wall clock as measured by the client, so they include queuing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

# Routed by the deterministic template path -- no LLM call.
DEFAULT_QUESTION = "各科室的就诊量是多少"


def _scrape_cache_events(base: str) -> dict:
    """Cache hit/miss counters from the service's own /metrics.

    Recorded so a saved run says for itself whether the cache was live, rather
    than leaving the reader to infer it from prose. Reading this process's
    `CACHE_ENABLED` would be wrong -- the bench is often launched from a
    different shell than the server.
    """
    events: dict[str, int] = {}
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5.0) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - metrics are diagnostic, never fatal
        return {"scraped": False}
    for line in text.splitlines():
        if line.startswith("medigraph_cache_events_total"):
            name, _, value = line.rpartition(" ")
            outcome = name.partition('outcome="')[2].partition('"')[0] or "unlabelled"
            try:
                events[outcome] = int(float(value))
            except ValueError:
                continue
    return {"scraped": True, "events": events,
            "cache_active": bool(events) and sum(events.values()) > 0}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def worker(
    base: str,
    path: str,
    payload: dict,
    count: int,
    latencies: list[float],
    statuses: list[int],
    lock: threading.Lock,
    timeout: float,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    for _ in range(count):
        request = urllib.request.Request(
            f"{base}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        status = 0
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        except Exception:  # noqa: BLE001 - connection errors count as failures
            status = 0
        elapsed = time.perf_counter() - started
        with lock:
            latencies.append(elapsed)
            statuses.append(status)


def main() -> None:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8020")
    parser.add_argument("--path", default="/api/v1/analysis/nl2sql")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    # The cache-off baseline and the cache-on run are two different measurements
    # of the same endpoint. With a hard-coded output path the second run silently
    # overwrote the first, leaving docs/PERFORMANCE.md §1 citing an artifact that
    # held §7's numbers.
    parser.add_argument("--out", default="", help="output json path (default outputs/bench_api_load.json)")
    args = parser.parse_args()

    payload = {"question": args.question}

    try:
        with urllib.request.urlopen(f"{args.base}/healthz", timeout=5) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"service not reachable at {args.base} ({exc}). Start it with: python service/main.py"
        )

    # Warm up so first-call lazy loading (schema linker vocabulary, DB handle) does
    # not land inside the measured window.
    warm_latencies: list[float] = []
    warm_statuses: list[int] = []
    worker(args.base, args.path, payload, args.warmup, warm_latencies, warm_statuses, threading.Lock(), args.timeout)
    print(f"warmup: {args.warmup} requests, first={warm_latencies[0] * 1000:.1f}ms last={warm_latencies[-1] * 1000:.1f}ms")

    per_worker = max(1, args.requests // args.concurrency)
    total = per_worker * args.concurrency
    latencies: list[float] = []
    statuses: list[int] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(args.base, args.path, payload, per_worker, latencies, statuses, lock, args.timeout),
            name=f"load-{index}",
        )
        for index in range(args.concurrency)
    ]

    print(f"load: {total} requests at concurrency {args.concurrency} -> {args.path}", flush=True)
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    ok = sum(1 for status in statuses if status == 200)
    report = {
        "base_url": args.base,
        "path": args.path,
        "question": args.question,
        "concurrency": args.concurrency,
        "requests": total,
        "successful": ok,
        "error_rate": round(1 - ok / total, 4) if total else 0.0,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(total / wall, 1) if wall else 0.0,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies) * 1000, 2),
            "p50": round(percentile(latencies, 0.50) * 1000, 2),
            "p90": round(percentile(latencies, 0.90) * 1000, 2),
            "p95": round(percentile(latencies, 0.95) * 1000, 2),
            "p99": round(percentile(latencies, 0.99) * 1000, 2),
            "max": round(max(latencies) * 1000, 2),
        },
        "status_counts": {str(code): statuses.count(code) for code in sorted(set(statuses))},
        # Scraped from the service under test, not from this process's own env:
        # the two differ whenever the bench is launched from a shell that does
        # not share the server's configuration, and it is the server's cache
        # state that the numbers actually reflect.
        "cache_events": _scrape_cache_events(args.base),
        "note": (
            "Deterministic template route: no LLM call, so this measures the service "
            "path (middleware, metrics, validation, schema linking, guarded SQLite "
            "execution). LLM-backed routes are bounded by provider latency, not by "
            "this path -- see bench_stream_ttft.json for those."
        ),
    }
    path = write_json(report, Path(args.out) if args.out
                      else Path(OUTPUTS_DIR) / "bench_api_load.json")

    print("\n================ HTTP LOAD ================")
    print(f"  {total} requests, concurrency {args.concurrency}, {ok} ok, {report['error_rate'] * 100:.1f}% errors")
    print(f"  throughput  {report['throughput_rps']} req/s")
    latency = report["latency_ms"]
    print(f"  latency ms  p50 {latency['p50']}  p90 {latency['p90']}  p95 {latency['p95']}  p99 {latency['p99']}  max {latency['max']}")
    print(f"  (saved -> {path})")


if __name__ == "__main__":
    main()
