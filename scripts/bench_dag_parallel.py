r"""Measure layered-parallel vs serial DAG execution on the real operator pipeline.

Runs the same DAG through `DAGExecutor(parallel=False)` and `parallel=True` and
reports wall-clock per node and end to end.

The linear ETL chain (load -> clean -> quality -> pii -> chunk -> ner -> link -> re
-> validate) has one node per layer, so parallelism cannot help it -- that is the
honest baseline. The speedup shows up where the DAG actually branches, so a second
DAG exercises independent branches over the same document.

    python scripts/bench_dag_parallel.py --max-docs 2
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CORPUS_DIR, OUTPUTS_DIR  # noqa: E402
from medigraph.agents.dag_executor import DAGExecutor, topological_layers  # noqa: E402
from medigraph.operators.base import get_operator, load_default_operators  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

# Linear pipeline: one node per layer, so it is the parallelism-immune control.
LINEAR_DAG = [
    {"id": "n1", "op": "text_clean", "deps": []},
    {"id": "n2", "op": "data_quality", "deps": ["n1"]},
    {"id": "n3", "op": "pii_redact", "deps": ["n2"]},
    {"id": "n4", "op": "chunker", "deps": ["n3"]},
    {"id": "n5", "op": "medical_ner", "deps": ["n4"]},
    {"id": "n6", "op": "entity_linker", "deps": ["n5"]},
    {"id": "n7", "op": "medical_re", "deps": ["n6"]},
    {"id": "n8", "op": "triple_validator", "deps": ["n7"]},
]

# Branching pipeline: quality profiling, PII redaction and chunking are independent
# given cleaned text, so they share a layer.
BRANCHED_DAG = [
    {"id": "n1", "op": "text_clean", "deps": []},
    {"id": "q", "op": "data_quality", "deps": ["n1"]},
    {"id": "p", "op": "pii_redact", "deps": ["n1"]},
    {"id": "c", "op": "chunker", "deps": ["n1"]},
    {"id": "ner", "op": "medical_ner", "deps": ["c"]},
    {"id": "link", "op": "entity_linker", "deps": ["ner"]},
    {"id": "re", "op": "medical_re", "deps": ["link"]},
    {"id": "val", "op": "triple_validator", "deps": ["re"]},
]


def load_text(max_docs: int, max_chars: int) -> str:
    """Load corpus text, truncated.

    The input is deliberately small: each run drives real LLM calls through NER/RE,
    and this benchmark measures *scheduling* (layer concurrency), not corpus
    throughput. A large input would multiply cost without changing the conclusion.
    """
    paths = sorted(Path(CORPUS_DIR).glob("*.txt"))[:max_docs]
    if not paths:
        raise SystemExit(f"no .txt documents under {CORPUS_DIR}")
    loader = get_operator("document_loader")
    out = loader.run({"paths": [str(path) for path in paths]})
    documents = out.get("documents", [])
    return "\n\n".join(document.get("text", "") for document in documents)[:max_chars]


class _LatencyShim:
    """Wraps a registered operator to add a fixed delay to every call.

    Why this exists: with the offline lexicon backend the whole pipeline runs in
    milliseconds, so scheduling changes are unmeasurable; with the LLM backend each
    node costs seconds but the provider's variance swamps the signal (and it burns
    quota). Injecting a fixed delay on the *real* DAG shapes isolates the scheduler:
    the delay stands in for the LLM latency that dominates production runs.

    Results from this mode are labelled `synthetic_latency_s` in the report and must
    not be presented as end-to-end pipeline timings.
    """

    def __init__(self, operator, seconds: float):
        self._operator = operator
        self._seconds = seconds
        self.meta = operator.meta

    def run(self, inputs: dict, **kwargs) -> dict:
        time.sleep(self._seconds)
        return self._operator.run(inputs, **kwargs)


def install_latency(dag: list[dict], seconds: float) -> None:
    from medigraph.operators.base import OP_REGISTRY

    for node in dag:
        name = node["op"]
        current = OP_REGISTRY[name]
        if not isinstance(current, _LatencyShim):
            OP_REGISTRY[name] = _LatencyShim(current, seconds)


def remove_latency() -> None:
    from medigraph.operators.base import OP_REGISTRY

    for name, operator in list(OP_REGISTRY.items()):
        if isinstance(operator, _LatencyShim):
            OP_REGISTRY[name] = operator._operator


def timed_run(dag: list[dict], payload: dict, parallel: bool, workers: int) -> dict:
    executor = DAGExecutor(max_retries=0, max_workers=workers, parallel=parallel)
    started = time.perf_counter()
    result = executor.run(dag, dict(payload), verbose=False)
    elapsed = time.perf_counter() - started
    report = result["report"]
    return {
        "seconds": round(elapsed, 3),
        "nodes_success": report["nodes_success"],
        "nodes_failed": report["nodes_failed"],
        "layers": report["layers"],
        "max_layer_width": report["max_layer_width"],
        "node_seconds": {
            nid: state.get("seconds") for nid, state in result["states"].items()
        },
    }


def main() -> None:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--simulate-latency",
        type=float,
        default=0.0,
        help=(
            "Add this many seconds to every operator call, standing in for LLM "
            "latency, to isolate the scheduler from provider variance."
        ),
    )
    args = parser.parse_args()

    load_default_operators()
    text = load_text(args.max_docs, args.max_chars)
    payload = {"text": text}
    print(f"input: {len(text)} chars from {args.max_docs} document(s)\n", flush=True)

    shapes = {"linear": LINEAR_DAG, "branched": BRANCHED_DAG}
    results: dict[str, dict] = {}

    if args.simulate_latency > 0:
        print(f"synthetic latency: +{args.simulate_latency}s per operator call\n", flush=True)

    for name, dag in shapes.items():
        layers = topological_layers(dag)
        widths = [len(layer) for layer in layers]
        print(f"[{name}] {len(dag)} nodes, {len(layers)} layers, widths={widths}")
        if args.simulate_latency > 0:
            install_latency(dag, args.simulate_latency)
        serial_times: list[float] = []
        parallel_times: list[float] = []
        detail: dict = {}
        for repeat in range(1, args.repeats + 1):
            serial = timed_run(dag, payload, parallel=False, workers=args.workers)
            parallel = timed_run(dag, payload, parallel=True, workers=args.workers)
            serial_times.append(serial["seconds"])
            parallel_times.append(parallel["seconds"])
            detail = {"serial": serial, "parallel": parallel}
            print(
                f"  [{repeat}] serial {serial['seconds']:7.2f}s  "
                f"parallel {parallel['seconds']:7.2f}s  "
                f"(success {parallel['nodes_success']}/{len(dag)})",
                flush=True,
            )
        serial_p50 = statistics.median(serial_times)
        parallel_p50 = statistics.median(parallel_times)
        speedup = round(serial_p50 / parallel_p50, 2) if parallel_p50 else None
        results[name] = {
            "nodes": len(dag),
            "layers": len(layers),
            "layer_widths": widths,
            "serial_p50_s": round(serial_p50, 3),
            "parallel_p50_s": round(parallel_p50, 3),
            "speedup_x": speedup,
            "saved_s": round(serial_p50 - parallel_p50, 3),
            "last_run": detail,
        }
        print(f"  -> p50 {serial_p50:.2f}s -> {parallel_p50:.2f}s  ({speedup}x)\n")
        remove_latency()

    report = {
        "input_chars": len(text),
        "documents": args.max_docs,
        "workers": args.workers,
        "repeats": args.repeats,
        "extraction_backend": __import__("os").getenv("EXTRACTION_BACKEND", "auto"),
        "synthetic_latency_s": args.simulate_latency,
        "shapes": results,
        "interpretation": (
            "The linear ETL chain has one node per layer, so layered parallelism "
            "cannot help it -- it is the control, and its speedup should be ~1.0x. "
            "The branched DAG puts quality/PII/chunking in one layer, which is where "
            "concurrency pays off; its ceiling is the layer width (3). "
            "With synthetic_latency_s = 0 and the offline lexicon backend the whole "
            "pipeline runs in milliseconds, so neither shape shows a difference: the "
            "scheduling win scales with per-node latency, which in production is "
            "dominated by LLM calls. Operators are IO-bound, which is why threads are "
            "the right primitive despite the GIL."
        ),
    }
    path = write_json(report, Path(OUTPUTS_DIR) / "bench_dag_parallel.json")
    print(f"(saved -> {path})")


if __name__ == "__main__":
    main()
