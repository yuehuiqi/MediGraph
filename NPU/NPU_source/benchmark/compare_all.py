"""Unified three-way comparison: CPU vs torch_npu vs custom Ascend C kernel.

One command produces the full comparison table for the report:
  - CPU (torch, fp32) and NPU (torch_npu) are timed in-process.
  - The custom Ascend C kernel is built+run via ../ascendc/run.sh and its
    RESULT_JSON line is parsed.

Run on the 910B3 (inside the CANN container, with the Ascend C toolkit for the
custom-kernel row):
  python NPU/benchmark/compare_all.py --rows 200000 --labels 32 --iters 200
Writes NPU/results/compare_all.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

# Suppress torch_npu startup warnings about CANN toolkit directory ownership.
warnings.filterwarnings("ignore", message=".*owner does not match.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ascend-toolkit.*", category=UserWarning)

import torch

HERE = Path(__file__).resolve().parent
ASCENDC = HERE.parent / "ascendc"
RESULTS = HERE.parent / "results"

# Reuse the timing routine from the Level-1 benchmark.
sys.path.insert(0, str(HERE))
from bench_softmax import _bench, _HAS_NPU  # noqa: E402


def run_ascendc(rows: int, labels: int, iters: int, blockdim: int) -> dict | None:
    """Build + run the custom Ascend C kernel; parse its RESULT_JSON line."""
    run_sh = ASCENDC / "run.sh"
    if not run_sh.exists():
        print("[compare_all] run.sh not found, skipping custom kernel.")
        return None
    try:
        out = subprocess.run(
            ["bash", str(run_sh), str(rows), str(labels), str(iters), str(blockdim)],
            cwd=str(ASCENDC), capture_output=True, text=True, timeout=1200,
        )
        text = out.stdout + "\n" + out.stderr
        m = re.search(r"RESULT_JSON\s+(\{.*\})", text)
        if m:
            return json.loads(m.group(1))
        # Print the full build+run output so the user can see the actual error.
        print("[compare_all] custom kernel: no RESULT_JSON found. Full output:")
        print("-" * 60)
        print(text[-3000:])  # last 3000 chars (cmake errors are near the end)
        print("-" * 60)
    except Exception as exc:  # noqa: BLE001
        print(f"[compare_all] Ascend C build/run skipped: {exc}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--labels", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--blockdim", type=int, default=40)
    ap.add_argument("--skip-ascendc", action="store_true", help="skip the custom-kernel row")
    args = ap.parse_args()

    torch.manual_seed(42)
    x = torch.randn(args.rows, args.labels, dtype=torch.float32)
    rows_table: list[dict] = []

    cpu = _bench("cpu", x, args.iters)
    cpu["system"] = "cpu_torch"
    rows_table.append(cpu)

    if _HAS_NPU:
        npu = _bench("npu:0", x, args.iters)
        npu["system"] = "npu_torch"
        rows_table.append(npu)

    if not args.skip_ascendc:
        ak = run_ascendc(args.rows, args.labels, args.iters, args.blockdim)
        if ak:
            rows_table.append(ak)

    base = cpu["throughput_rows_per_s"] or 1.0
    print(f"\nshape=[{args.rows},{args.labels}] iters={args.iters}\n")
    print(f"| {'system':16s} | throughput(rows/s) | P50(ms) | P99(ms) | speedup |")
    print(f"| {'-'*16} | --- | --- | --- | --- |")
    for r in rows_table:
        sp = r["throughput_rows_per_s"] / base
        print(f"| {r['system']:16s} | {r['throughput_rows_per_s']:.0f} | "
              f"{r.get('latency_p50_ms','-')} | {r.get('latency_p99_ms','-')} | {sp:.2f}x |")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compare_all.json").write_text(
        json.dumps({"shape": [args.rows, args.labels], "iters": args.iters, "results": rows_table},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS / 'compare_all.json'}")
    if not _HAS_NPU:
        print("[!] NPU rows absent here — run inside the 910B2 CANN container for the full table.")


if __name__ == "__main__":
    main()
