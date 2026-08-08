"""Level 1 benchmark: medical entity-scoring Softmax on CPU vs Ascend NPU.

The NER decode stage normalizes a score matrix [N_spans, C_labels] row-wise with
softmax (then argmax/top-k). This is a real, compute-dense, regular-shape hot spot
-- a fair CPU/NPU comparison. This script needs only torch + torch_npu (present in
the mindspeed/CANN image); no custom-op compiler required.

Reports the "three-piece" metrics: throughput (rows/s), latency (P50/P99), and
energy efficiency (rows/Joule via npu-smi power sampling).

Run on the 910B3 container:
  python NPU/benchmark/bench_softmax.py --rows 200000 --labels 32 --iters 200
Writes NPU/results/bench_softmax.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path

# Suppress torch_npu startup warnings about CANN toolkit directory ownership.
# These are harmless permission-metadata mismatches in shared/container envs.
warnings.filterwarnings("ignore", message=".*owner does not match.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ascend-toolkit.*", category=UserWarning)

import torch

try:
    import torch_npu  # noqa: F401  (registers the 'npu' device)
    _HAS_NPU = torch.npu.is_available()
except Exception:  # noqa: BLE001
    _HAS_NPU = False

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _bench(
    device: str,
    x: torch.Tensor,
    iters: int,
    warmup: int = 20,
    latency_iters: int | None = None,
) -> dict:
    """Measure sustained throughput and single-request latency separately."""
    xd = x.to(device)
    sync = (
        torch.npu.synchronize
        if device.startswith("npu")
        else torch.cuda.synchronize
        if device.startswith("cuda")
        else lambda: None
    )
    latency_iters = latency_iters or iters

    with torch.no_grad():
        for _ in range(warmup):
            output = torch.softmax(xd, dim=-1)
        sync()

        # Sustained throughput: enqueue all work, synchronize once.
        throughput_start = time.perf_counter()
        for _ in range(iters):
            output = torch.softmax(xd, dim=-1)
        sync()
        total = time.perf_counter() - throughput_start

        # Single-request latency: synchronize every request.
        latencies = []
        for _ in range(latency_iters):
            start = time.perf_counter()
            output = torch.softmax(xd, dim=-1)
            sync()
            latencies.append(time.perf_counter() - start)

    rows = x.shape[0]
    ordered = sorted(latencies)

    return {
        "device": device,
        "rows": rows,
        "iters": iters,
        "throughput_mode": "async_batch",
        "latency_mode": "launch_plus_sync",
        "total_s": round(total, 6),
        "throughput_rows_per_s": round(rows * iters / total, 1),
        "latency_p50_ms": round(
            ordered[len(ordered) // 2] * 1000, 6
        ),
        "latency_p95_ms": round(
            ordered[min(
                len(ordered) - 1,
                int(len(ordered) * 0.95)
            )] * 1000, 6
        ),
        "latency_p99_ms": round(
            ordered[min(
                len(ordered) - 1,
                int(len(ordered) * 0.99)
            )] * 1000, 6
        ),
        "latency_mean_ms": round(
            statistics.mean(latencies) * 1000, 6
        ),
    }


def _sample_npu_power(stop_after: float) -> float:
    """Average NPU power (W) sampled via npu-smi during `stop_after` seconds."""
    import subprocess, re
    readings = []
    end = time.time() + stop_after
    while time.time() < end:
        try:
            out = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
            for m in re.finditer(r"(\d+\.\d+)\s*/\s*\d+\.\d+\s*$", out, flags=re.MULTILINE):
                readings.append(float(m.group(1)))
            # fallback: any 'xx.x W' pattern
            if not readings:
                readings += [float(x) for x in re.findall(r"(\d{2,3}\.\d)\s", out)][:1]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return round(sum(readings) / len(readings), 1) if readings else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200000, help="candidate spans")
    ap.add_argument("--labels", type=int, default=32, help="entity-type labels (C)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--energy", action="store_true", help="also sample npu-smi power (rough)")
    args = ap.parse_args()

    torch.manual_seed(42)
    x = torch.randn(args.rows, args.labels, dtype=torch.float32)
    print(f"Score matrix: [{args.rows}, {args.labels}]  iters={args.iters}")
    print(f"NPU available: {_HAS_NPU}\n")

    report = {"shape": [args.rows, args.labels], "iters": args.iters, "results": {}}

    cpu = _bench("cpu", x, args.iters)
    report["results"]["cpu"] = cpu
    print(f"CPU : {cpu['throughput_rows_per_s']:.0f} rows/s  P50={cpu['latency_p50_ms']}ms  P99={cpu['latency_p99_ms']}ms")

    if _HAS_NPU:
        npu = _bench("npu:0", x, args.iters)
        report["results"]["npu"] = npu
        speedup = npu["throughput_rows_per_s"] / cpu["throughput_rows_per_s"] if cpu["throughput_rows_per_s"] else 0
        report["speedup_npu_vs_cpu"] = round(speedup, 2)
        print(f"NPU : {npu['throughput_rows_per_s']:.0f} rows/s  P50={npu['latency_p50_ms']}ms  P99={npu['latency_p99_ms']}ms")
        print(f"Speedup (NPU/CPU): {speedup:.2f}x")

        # correctness vs CPU
        with torch.no_grad():
            ref = torch.softmax(x, dim=-1)
            got = torch.softmax(x.to("npu:0"), dim=-1).cpu()
            max_abs = (ref - got).abs().max().item()
        report["max_abs_diff_vs_cpu"] = max_abs
        print(f"Correctness max|Δ| vs CPU: {max_abs:.2e}")

        if args.energy:
            print("Sampling NPU power for ~15s under load (rough energy estimate)...")
            # run a sustained load while sampling
            import threading
            stop = {"v": False}

            def load():
                xd = x.to("npu:0")
                while not stop["v"]:
                    torch.softmax(xd, dim=-1)
                    torch.npu.synchronize()

            th = threading.Thread(target=load, daemon=True)
            th.start()
            avg_w = _sample_npu_power(15)
            stop["v"] = True
            th.join(timeout=2)
            report["npu_avg_power_w"] = avg_w
            if avg_w:
                rows_per_joule = npu["throughput_rows_per_s"] / avg_w
                report["npu_rows_per_joule"] = round(rows_per_joule, 1)
                print(f"NPU avg power: {avg_w} W  ->  {rows_per_joule:.0f} rows/Joule")
    else:
        print("\n[!] torch_npu / NPU not available here. Run inside the CANN container on the 910B2.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bench_softmax.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS / 'bench_softmax.json'}")


if __name__ == "__main__":
    main()
