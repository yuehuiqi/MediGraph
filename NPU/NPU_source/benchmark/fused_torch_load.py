import argparse
import json
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=200000)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--temperature", type=float, default=1.35)
args = parser.parse_args()

torch.manual_seed(42)

scores = torch.randn(
    args.rows, 32, dtype=torch.float32
).npu()

mask_cpu = np.zeros(32, dtype=np.float32)
mask_cpu[::7] = -10000.0
mask = torch.from_numpy(mask_cpu).npu()

inverse_temperature = 1.0 / args.temperature

for _ in range(20):
    output = torch.softmax(
        scores * inverse_temperature + mask,
        dim=-1,
    )
torch.npu.synchronize()

print("LOAD_READY", flush=True)

launches = 0
start = time.perf_counter()

while True:
    for _ in range(200):
        output = torch.softmax(
            scores * inverse_temperature + mask,
            dim=-1,
        )
        launches += 1

    torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    if elapsed >= args.seconds:
        break

print(
    "LOAD_JSON",
    json.dumps({
        "system": "npu_torch_composed",
        "launches": launches,
        "seconds": elapsed,
        "rows_per_s": args.rows * launches / elapsed,
    }),
    flush=True,
)
