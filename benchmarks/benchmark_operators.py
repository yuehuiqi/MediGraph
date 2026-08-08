"""Operator performance benchmark: latency & throughput (CPU baseline).

Measures each operator over repeated runs and reports latency percentiles and
throughput. Directly supports Task-2 scoring ("算子处理延时低") and the
"performance quantification" encouraged direction. The NPU column is left as a
documented future comparison (out of current scope).

Usage:
  python benchmarks/benchmark_operators.py --samples 8
Writes outputs/benchmark_operators.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, RAW_DEMO_DIR  # noqa: E402
from medigraph.llm.client import LLMClient  # noqa: E402
from medigraph.operators.base import load_default_operators, get_operator  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import iter_documents, write_json  # noqa: E402

enable_utf8()

_SAMPLE_TEXT = (
    "Pheochromocytoma is a tumor of the adrenal medulla, positive for chromogranin "
    "and S100, associated with RET and VHL mutations; patients present with headaches, "
    "sweating and tachycardia. Treatment is adrenalectomy."
)


def _percentiles(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50": 0, "p95": 0, "mean": 0}
    s = sorted(latencies)
    p50 = s[len(s) // 2]
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    return {"p50_ms": round(p50 * 1000, 1), "p95_ms": round(p95 * 1000, 1),
            "mean_ms": round(statistics.mean(s) * 1000, 1)}


def _bench(op, inputs: dict, n: int) -> dict:
    lat = []
    for _ in range(n):
        t0 = time.time()
        op.run(inputs)
        lat.append(time.time() - t0)
    pct = _percentiles(lat)
    total = sum(lat)
    return {**pct, "runs": n, "throughput_samples_per_s": round(n / total, 2) if total else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8, help="runs per operator")
    args = ap.parse_args()

    llm = LLMClient()
    load_default_operators(llm=llm)
    print(f"Benchmarking operators (model={llm.config.model}, runs={args.samples}) ...\n")

    # CPU-only operators over a real document.
    docs = iter_documents(RAW_DEMO_DIR)
    doc_text = docs[0]["text"] if docs else _SAMPLE_TEXT
    results: dict[str, dict] = {}

    print("  text_clean ...")
    results["text_clean"] = {
        **_bench(get_operator("text_clean"), {"text": doc_text}, args.samples),
        "backend": "cpu", "type": "rule",
    }
    cleaned = get_operator("text_clean").run({"text": doc_text})["text"]

    print("  chunker ...")
    results["chunker"] = {
        **_bench(get_operator("chunker"), {"text": cleaned}, args.samples),
        "backend": "cpu", "type": "rule",
    }

    # LLM operators over a single representative chunk (fewer runs to bound cost).
    llm_runs = max(3, args.samples // 2)
    print(f"  medical_ner ... ({llm_runs} runs)")
    results["medical_ner"] = {
        **_bench(get_operator("medical_ner"), {"text": _SAMPLE_TEXT}, llm_runs),
        "backend": "cpu+API", "type": "llm",
    }
    ents = get_operator("medical_ner").run({"text": _SAMPLE_TEXT})["entities"]

    print(f"  medical_re ... ({llm_runs} runs)")
    results["medical_re"] = {
        **_bench(get_operator("medical_re"), {"text": _SAMPLE_TEXT, "entities": ents}, llm_runs),
        "backend": "cpu+API", "type": "llm",
    }
    tri = get_operator("medical_re").run({"text": _SAMPLE_TEXT, "entities": ents})["triples"]

    print("  triple_validator ...")
    results["triple_validator"] = {
        **_bench(get_operator("triple_validator"), {"triples": tri * 20}, args.samples),
        "backend": "cpu", "type": "rule",
        "note": f"{len(tri) * 20} triples/run",
    }

    write_json({"model": llm.config.model, "operators": results, "llm_stats": llm.stats.summary()},
               OUTPUTS_DIR / "benchmark_operators.json")

    # Markdown table
    print("\n| operator | type | backend | p50 (ms) | p95 (ms) | throughput (samples/s) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        print(f"| {name} | {r['type']} | {r['backend']} | {r['p50_ms']} | {r['p95_ms']} | {r['throughput_samples_per_s']} |")
    print(f"\n(saved -> {OUTPUTS_DIR / 'benchmark_operators.json'})")
    print("Note: NPU column is a documented future comparison (Ascend C), out of current scope.")


if __name__ == "__main__":
    main()
