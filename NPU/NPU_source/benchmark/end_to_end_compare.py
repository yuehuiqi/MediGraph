from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.npu_softmax_operator import AscendCMedicalSoftmax


def summarize(values):
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": ordered[len(ordered) // 2],
        "p99_ms": ordered[
            min(len(ordered) - 1, int(len(ordered) * 0.99))
        ],
    }


rows, labels, iterations = 200000, 32, 200
rng = np.random.default_rng(42)
scores = rng.normal(size=(rows, labels)).astype(np.float32)

custom = AscendCMedicalSoftmax(block_dim=40)

for _ in range(5):
    custom.softmax(scores)

custom_times = []
for _ in range(iterations):
    start = time.perf_counter()
    custom_output = custom.softmax(scores)
    custom_times.append((time.perf_counter() - start) * 1000)

for _ in range(5):
    tensor = torch.from_numpy(scores).to("npu:0")
    output = torch.softmax(tensor, dim=-1).cpu()
torch.npu.synchronize()

torch_times = []
for _ in range(iterations):
    start = time.perf_counter()
    tensor = torch.from_numpy(scores).to("npu:0")
    output = torch.softmax(tensor, dim=-1).cpu()
    torch.npu.synchronize()
    torch_times.append((time.perf_counter() - start) * 1000)

reference = np.exp(scores - scores.max(axis=1, keepdims=True))
reference /= reference.sum(axis=1, keepdims=True)

report = {
    "shape": [rows, labels],
    "scope": "pageable_H2D + kernel + D2H",
    "results": {
        "npu_torch": summarize(torch_times),
        "npu_ascendc": summarize(custom_times),
    },
    "ascendc_max_abs_diff": float(
        np.max(np.abs(custom_output - reference))
    ),
}

path = ROOT / "results" / "end_to_end_compare.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print("Saved ->", path)
