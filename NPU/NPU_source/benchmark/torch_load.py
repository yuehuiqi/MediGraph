import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch_npu  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=200000)
parser.add_argument("--labels", type=int, default=32)
parser.add_argument("--seconds", type=float, default=20)
args = parser.parse_args()

x = torch.randn(
    args.rows, args.labels,
    dtype=torch.float32,
    device="npu:0",
)

with torch.no_grad():
    for _ in range(20):
        y = torch.softmax(x, dim=-1)
    torch.npu.synchronize()

    print("LOAD_READY", flush=True)

    launches = 0
    start = time.perf_counter()

    while time.perf_counter() - start < args.seconds:
        for _ in range(128):
            y = torch.softmax(x, dim=-1)
        torch.npu.synchronize()
        launches += 128

    elapsed = time.perf_counter() - start

result = {
    "system": "npu_torch",
    "launches": launches,
    "seconds": elapsed,
    "rows_per_s": args.rows * launches / elapsed,
}
print("LOAD_JSON " + json.dumps(result), flush=True)
