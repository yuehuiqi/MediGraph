r"""Measure the streaming win: time-to-first-token vs blocking total latency.

Runs the same questions through `POST /api/v1/kg/qa` (blocking JSON) and
`POST /api/v1/kg/qa/stream` (SSE), and reports for each:

  * blocking  -- the user sees nothing until the whole answer is composed;
  * streaming -- TTFT, i.e. when the first visible character arrives.

Total generation time is essentially unchanged by streaming; the point is
perceived latency, so the two are reported separately rather than as one number.

    python scripts/bench_stream_ttft.py --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

QUESTIONS = [
    "高血压有哪些并发症？",
    "2型糖尿病推荐使用哪些药物？",
    "冠心病的临床表现有哪些？",
]


def post_json(base: str, path: str, payload: dict, timeout: float) -> tuple[dict, float]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, time.perf_counter() - started


def post_sse(base: str, path: str, payload: dict, timeout: float) -> dict:
    """Consume an SSE response, timing the first `delta` frame."""
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    ttfb: float | None = None  # first frame of any kind (the `meta` handshake)
    ttft: float | None = None  # first answer token
    deltas = 0
    text_parts: list[str] = []
    done: dict = {}
    event = ""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                if ttfb is None:
                    ttfb = time.perf_counter() - started
                data = json.loads(line[6:])
                if event == "delta":
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    deltas += 1
                    text_parts.append(data.get("text", ""))
                elif event == "done":
                    done = data
                elif event == "error":
                    done = {"error": data.get("error")}
    total = time.perf_counter() - started
    return {
        "ttfb_s": ttfb,
        "ttft_s": ttft,
        "total_s": total,
        "deltas": deltas,
        "chars": len("".join(text_parts)),
        "server_ttft_ms": done.get("ttft_ms"),
        "refused": done.get("refused"),
        "error": done.get("error"),
    }


def summarize(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    ordered = sorted(clean)
    return {
        "n": len(ordered),
        "mean_s": round(statistics.fmean(ordered), 3),
        "p50_s": round(ordered[len(ordered) // 2], 3),
        "min_s": round(ordered[0], 3),
        "max_s": round(ordered[-1], 3),
    }


def main() -> None:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8020")
    parser.add_argument("--graph", default="graph.json")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(f"{args.base}/healthz", timeout=5) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"service not reachable at {args.base} ({exc}). "
            "Start it with: python service/main.py"
        )

    blocking_totals: list[float] = []
    stream_ttfbs: list[float] = []
    stream_ttfts: list[float] = []
    stream_totals: list[float] = []
    rows: list[dict] = []

    for repeat in range(1, args.repeats + 1):
        for question in QUESTIONS:
            payload = {"question": question, "graph_name": args.graph, "hops": args.hops}

            body, blocking_total = post_json(args.base, "/api/v1/kg/qa", payload, args.timeout)
            blocking_totals.append(blocking_total)

            stream = post_sse(args.base, "/api/v1/kg/qa/stream", payload, args.timeout)
            if stream["ttft_s"] is not None:
                stream_ttfts.append(stream["ttft_s"])
            if stream["ttfb_s"] is not None:
                stream_ttfbs.append(stream["ttfb_s"])
            stream_totals.append(stream["total_s"])

            rows.append(
                {
                    "repeat": repeat,
                    "question": question,
                    "blocking_total_s": round(blocking_total, 3),
                    "blocking_chars": len(body.get("answer", "")),
                    "stream_ttfb_s": round(stream["ttfb_s"], 4) if stream["ttfb_s"] else None,
                    "stream_ttft_s": round(stream["ttft_s"], 3) if stream["ttft_s"] else None,
                    "stream_total_s": round(stream["total_s"], 3),
                    "stream_deltas": stream["deltas"],
                    "stream_chars": stream["chars"],
                    "refused": stream.get("refused"),
                }
            )
            print(
                f"  [{repeat}] {question[:22]:24} "
                f"blocking {blocking_total:6.2f}s | "
                f"stream TTFB {stream['ttfb_s'] or 0:6.3f}s "
                f"TTFT {stream['ttft_s'] or float('nan'):5.2f}s "
                f"total {stream['total_s']:6.2f}s "
                f"({stream['deltas']} deltas)",
                flush=True,
            )

    blocking = summarize(blocking_totals)
    ttfb = summarize(stream_ttfbs)
    ttft = summarize(stream_ttfts)
    report = {
        "base_url": args.base,
        "graph": args.graph,
        "hops": args.hops,
        "questions": len(QUESTIONS),
        "repeats": args.repeats,
        "blocking_time_to_first_output": blocking,
        "streaming_time_to_first_byte": ttfb,
        "streaming_time_to_first_token": ttft,
        "streaming_total": summarize(stream_totals),
        "improvement": (
            {
                "first_token_p50_speedup_x": round(blocking["p50_s"] / ttft["p50_s"], 2),
                "first_token_p50_saved_s": round(blocking["p50_s"] - ttft["p50_s"], 2),
                "first_byte_p50_speedup_x": (
                    round(blocking["p50_s"] / ttfb["p50_s"], 1) if ttfb.get("p50_s") else None
                ),
            }
            if blocking and ttft and ttft["p50_s"]
            else {}
        ),
        "interpretation": (
            "Blocking shows nothing until the answer is complete, so its total latency "
            "IS its time-to-first-output. Streaming leaves total generation time "
            "unchanged and improves perceived latency in two steps: the SSE `meta` "
            "frame arrives in milliseconds (time-to-first-byte, so the UI can render "
            "immediately), and answer tokens start at time-to-first-token. The "
            "remaining first-token gap is dominated by the retrieval phase, which runs "
            "a blocking LLM NER call on the question plus graph traversal before "
            "composition begins -- that is the next optimisation target, not streaming."
        ),
        "detail": rows,
    }
    path = write_json(report, Path(OUTPUTS_DIR) / "bench_stream_ttft.json")

    print("\n================ PERCEIVED LATENCY ================")
    print(f"  blocking, first output    p50 {blocking.get('p50_s')}s   (n={blocking.get('n')})")
    print(f"  streaming, first byte     p50 {ttfb.get('p50_s')}s   (n={ttfb.get('n')})")
    print(f"  streaming, first token    p50 {ttft.get('p50_s')}s   (n={ttft.get('n')})")
    improvement = report["improvement"]
    if improvement:
        print(
            f"  -> first token {improvement['first_token_p50_speedup_x']}x sooner "
            f"({improvement['first_token_p50_saved_s']}s)"
        )
        if improvement.get("first_byte_p50_speedup_x"):
            print(f"  -> first byte  {improvement['first_byte_p50_speedup_x']}x sooner")
    print(f"  (saved -> {path})")


if __name__ == "__main__":
    main()
