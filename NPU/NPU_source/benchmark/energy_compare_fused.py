from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=200000)
parser.add_argument("--idle-seconds", type=float, default=10)
parser.add_argument("--load-seconds", type=float, default=20)
parser.add_argument("--interval", type=float, default=0.15)
parser.add_argument("--temperature", type=float, default=1.35)
args = parser.parse_args()


def read_power():
    result = subprocess.run(
        ["npu-smi", "info"],
        text=True,
        capture_output=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        fields = [
            item.strip()
            for item in line.split("|")
            if item.strip()
        ]

        if not fields or "910" not in fields[0]:
            continue

        if len(fields) >= 3:
            match = re.search(
                r"([0-9]+(?:\.[0-9]+)?)",
                fields[2],
            )
            if match:
                value = float(match.group(1))
                if 10 <= value <= 1000:
                    return value

    raise RuntimeError(
        "cannot parse power from npu-smi:\n"
        + result.stdout
    )


def sample_idle(seconds):
    samples = []
    deadline = time.perf_counter() + seconds

    while time.perf_counter() < deadline:
        samples.append(read_power())
        time.sleep(args.interval)

    return samples


def run_load(command):
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    ready = False

    while True:
        line = process.stdout.readline()

        if line:
            print(line, end="")
            lines.append(line.rstrip())
            if line.strip() == "LOAD_READY":
                ready = True
                break
        elif process.poll() is not None:
            break

    if not ready:
        process.wait()
        raise RuntimeError(
            "load process exited before LOAD_READY"
        )

    powers = []

    while process.poll() is None:
        powers.append(read_power())
        time.sleep(args.interval)

    remaining = process.stdout.read()
    if remaining:
        print(remaining, end="")
        lines.extend(remaining.splitlines())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"load process failed: {return_code}"
        )

    load_json = None

    for line in lines:
        if "LOAD_JSON " in line:
            load_json = json.loads(
                line.split("LOAD_JSON ", 1)[1]
            )

    if load_json is None:
        raise RuntimeError("LOAD_JSON not found")
    if not powers:
        raise RuntimeError("no power samples collected")

    return load_json, powers


print("Measuring idle power...")
idle_samples = sample_idle(args.idle_seconds)
idle_mean = statistics.mean(idle_samples)
print(f"idle mean: {idle_mean:.2f} W")

torch_command = [
    sys.executable,
    str(ROOT / "benchmark/fused_torch_load.py"),
    "--rows", str(args.rows),
    "--seconds", str(args.load_seconds),
    "--temperature", str(args.temperature),
]

custom_command = [
    str(ROOT / "ascendc/run_fused.sh"),
    str(args.rows),
    "32",
    "20",
    "40",
    "20",
    str(args.temperature),
    str(args.load_seconds),
]

print("\nMeasuring torch_npu composed fusion...")
torch_load, torch_power = run_load(torch_command)

print("\nMeasuring AscendC fused kernel...")
custom_load, custom_power = run_load(custom_command)


def make_result(load, powers):
    mean_power = statistics.mean(powers)
    dynamic_power = mean_power - idle_mean
    throughput = load["rows_per_s"]

    return {
        "throughput_rows_per_s": throughput,
        "mean_power_w": mean_power,
        "median_power_w": statistics.median(powers),
        "dynamic_power_w": dynamic_power,
        "gross_rows_per_joule": throughput / mean_power,
        "dynamic_rows_per_joule":
            throughput / max(dynamic_power, 0.001),
        "power_samples": len(powers),
    }


torch_result = make_result(torch_load, torch_power)
custom_result = make_result(custom_load, custom_power)

report = {
    "operation": "temperature+label_mask+softmax",
    "shape": [args.rows, 32],
    "temperature": args.temperature,
    "idle_mean_power_w": idle_mean,
    "systems": {
        "npu_torch_composed": torch_result,
        "npu_ascendc_fused": custom_result,
    },
    "speedup": {
        "throughput":
            custom_result["throughput_rows_per_s"]
            / torch_result["throughput_rows_per_s"],
        "gross_energy_efficiency":
            custom_result["gross_rows_per_joule"]
            / torch_result["gross_rows_per_joule"],
        "dynamic_energy_efficiency":
            custom_result["dynamic_rows_per_joule"]
            / torch_result["dynamic_rows_per_joule"],
    },
}

print("\n| system | throughput | mean W | gross rows/J | dynamic rows/J |")
print("|---|---:|---:|---:|---:|")

for name, result in report["systems"].items():
    print(
        f"| {name} "
        f"| {result['throughput_rows_per_s']:.0f} "
        f"| {result['mean_power_w']:.1f} "
        f"| {result['gross_rows_per_joule']:.0f} "
        f"| {result['dynamic_rows_per_joule']:.0f} |"
    )

path = ROOT / "results/fused_energy_compare.json"
path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report["speedup"], indent=2))
print("Saved ->", path)
