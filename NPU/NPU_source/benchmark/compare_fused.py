from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.npu_fused_operator import (  # noqa: E402
    AscendCFusedMedicalSoftmax,
)


def summarize(values):
    ordered = sorted(values)

    def percentile(q):
        index = max(0, int(np.ceil(len(ordered) * q)) - 1)
        return ordered[min(index, len(ordered) - 1)]

    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def cv(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / statistics.mean(values) * 100


parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=200000)
parser.add_argument("--labels", type=int, default=32)
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--latency-iters", type=int, default=200)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--blockdim", type=int, default=40)
parser.add_argument("--temperature", type=float, default=1.35)
args = parser.parse_args()

if args.labels != 32:
    raise SystemExit("optimized path requires labels=32")

torch.manual_seed(42)

scores = torch.randn(
    args.rows,
    args.labels,
    dtype=torch.float32,
).npu()

mask_cpu = np.zeros(args.labels, dtype=np.float32)
mask_cpu[::7] = -10000.0

mask = torch.from_numpy(mask_cpu).npu()
custom_output = torch.empty_like(scores)

custom = AscendCFusedMedicalSoftmax(
    block_dim=args.blockdim
)

inverse_temperature = 1.0 / args.temperature

for _ in range(20):
    torch_output = torch.softmax(
        scores * inverse_temperature + mask,
        dim=-1,
    )
torch.npu.synchronize()

torch.npu.synchronize()
for _ in range(20):
    custom.enqueue_device(
        scores, mask, custom_output, args.temperature
    )
custom.synchronize()

torch_throughputs = []
custom_throughputs = []
torch_latency_runs = []
custom_latency_runs = []

for repeat in range(args.repeats):
    torch.npu.synchronize()
    start = time.perf_counter()

    for _ in range(args.iters):
        torch_output = torch.softmax(
            scores * inverse_temperature + mask,
            dim=-1,
        )

    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    torch_throughputs.append(
        args.rows * args.iters / elapsed
    )

    torch.npu.synchronize()
    start = time.perf_counter()

    for _ in range(args.iters):
        custom.enqueue_device(
            scores,
            mask,
            custom_output,
            args.temperature,
        )

    custom.synchronize()
    elapsed = time.perf_counter() - start

    custom_throughputs.append(
        args.rows * args.iters / elapsed
    )

    torch_latencies = []
    for _ in range(args.latency_iters):
        start = time.perf_counter()
        torch_output = torch.softmax(
            scores * inverse_temperature + mask,
            dim=-1,
        )
        torch.npu.synchronize()
        torch_latencies.append(
            (time.perf_counter() - start) * 1000
        )

    custom_latencies = []
    for _ in range(args.latency_iters):
        start = time.perf_counter()
        custom.enqueue_device(
            scores,
            mask,
            custom_output,
            args.temperature,
        )
        custom.synchronize()
        custom_latencies.append(
            (time.perf_counter() - start) * 1000
        )

    torch_latency_runs.append(summarize(torch_latencies))
    custom_latency_runs.append(summarize(custom_latencies))

    print(
        f"run={repeat + 1} "
        f"torch={torch_throughputs[-1]:.0f} "
        f"ascendc={custom_throughputs[-1]:.0f}"
    )

custom.synchronize()

expected = torch.softmax(
    scores * inverse_temperature + mask,
    dim=-1,
)
torch.npu.synchronize()

expected_cpu = expected.cpu().numpy()
actual_cpu = custom_output.cpu().numpy()

max_abs_diff = float(
    np.max(np.abs(actual_cpu - expected_cpu))
)
row_sum_error = float(
    np.max(np.abs(actual_cpu.sum(axis=1) - 1.0))
)
masked_probability_max = float(
    np.max(actual_cpu[:, ::7])
)

torch_median = statistics.median(torch_throughputs)
custom_median = statistics.median(custom_throughputs)

torch_p50 = statistics.median(
    row["p50_ms"] for row in torch_latency_runs
)
torch_p99 = statistics.median(
    row["p99_ms"] for row in torch_latency_runs
)
custom_p50 = statistics.median(
    row["p50_ms"] for row in custom_latency_runs
)
custom_p99 = statistics.median(
    row["p99_ms"] for row in custom_latency_runs
)

report = {
    "operation": "temperature+label_mask+softmax",
    "shape": [args.rows, args.labels],
    "temperature": args.temperature,
    "repeats": args.repeats,
    "iterations": args.iters,
    "scope": "device_resident",
    "results": {
        "npu_torch_composed": {
            "throughput_median_rows_per_s": torch_median,
            "throughput_cv_percent": cv(torch_throughputs),
            "p50_median_ms": torch_p50,
            "p99_median_ms": torch_p99,
        },
        "npu_ascendc_fused": {
            "throughput_median_rows_per_s": custom_median,
            "throughput_cv_percent": cv(custom_throughputs),
            "p50_median_ms": custom_p50,
            "p99_median_ms": custom_p99,
        },
    },
    "speedup": {
        "throughput": custom_median / torch_median,
        "p50": torch_p50 / custom_p50,
        "p99": torch_p99 / custom_p99,
    },
    "correctness": {
        "max_abs_diff": max_abs_diff,
        "row_sum_error": row_sum_error,
        "masked_probability_max": masked_probability_max,
    },
}

path = ROOT / "results/fused_compare_repeated.json"
path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print("Saved ->", path)

assert max_abs_diff <= 1e-5
assert row_sum_error <= 1e-5
assert masked_probability_max <= 1e-6
