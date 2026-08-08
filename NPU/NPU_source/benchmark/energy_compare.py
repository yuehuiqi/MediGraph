from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_power() -> float | None:
    try:
        output = subprocess.check_output(
            ["npu-smi", "info"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )

        # Current format:
        # | 0  910B3 | OK | 98.0  50  0 / 0 |
        for line in output.splitlines():
            match = re.search(
                r"^\|\s*\d+\s+\S*910\S*\s+"
                r"\|\s*\S+\s+\|\s*"
                r"(\d+(?:\.\d+)?)",
                line,
            )
            if match:
                value = float(match.group(1))
                if 20 <= value <= 1000:
                    return value

        # Compatibility with current/rated format.
        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?",
            output,
        )
        for raw in matches:
            value = float(raw)
            if 20 <= value <= 1000:
                return value

        return None
    except Exception:
        return None


def sample_power(seconds: float, interval: float = 0.5) -> dict:
    readings = []
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        value = read_power()
        if value is not None:
            readings.append(value)
        time.sleep(interval)

    if not readings:
        raise RuntimeError("No power reading parsed from npu-smi")

    ordered = sorted(readings)
    return {
        "samples": len(readings),
        "mean_w": sum(readings) / len(readings),
        "min_w": ordered[0],
        "max_w": ordered[-1],
        "p50_w": ordered[len(ordered) // 2],
    }


def run_load(command: list[str], ready_token: str) -> tuple[dict, dict]:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    ready = False

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
        if ready_token in line:
            ready = True
            break

    if not ready:
        process.kill()
        raise RuntimeError(
            f"Load process exited before {ready_token}"
        )

    power = sample_power(15)

    remaining, _ = process.communicate(timeout=60)
    print(remaining, end="")
    text = "".join(output_lines) + remaining

    match = re.search(r"LOAD_JSON\s+(\{.*\})", text)
    if not match:
        raise RuntimeError("LOAD_JSON not found")

    return json.loads(match.group(1)), power


def efficiency(load: dict, power: dict, idle_w: float) -> dict:
    throughput = float(load["rows_per_s"])
    mean_w = float(power["mean_w"])
    dynamic_w = max(mean_w - idle_w, 1e-9)

    return {
        **load,
        "power": power,
        "gross_rows_per_joule": throughput / mean_w,
        "dynamic_rows_per_joule": throughput / dynamic_w,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Measuring idle power...")
    idle = sample_power(5)
    idle_w = idle["mean_w"]
    print(f"idle mean: {idle_w:.2f} W")

    print("\nMeasuring torch_npu...")
    torch_load, torch_power = run_load(
        [
            "python3",
            str(ROOT / "benchmark" / "torch_load.py"),
            "--rows", "200000",
            "--labels", "32",
            "--seconds", "20",
        ],
        "LOAD_READY",
    )

    print("\nMeasuring AscendC...")
    ascendc_load, ascendc_power = run_load(
        [
            str(
                ROOT / "ascendc" / "build"
                / "medical_softmax_run"
            ),
            "200000", "32", "20", "40", "20", "20",
        ],
        "LOAD_READY",
    )
    ascendc_load["system"] = "npu_ascendc"

    report = {
        "idle_power": idle,
        "results": [
            efficiency(torch_load, torch_power, idle_w),
            efficiency(ascendc_load, ascendc_power, idle_w),
        ],
    }

    path = RESULTS / "energy_compare.json"
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n| system | throughput | mean W | rows/J |")
    print("|---|---:|---:|---:|")
    for row in report["results"]:
        print(
            f"| {row['system']} "
            f"| {row['rows_per_s']:.0f} "
            f"| {row['power']['mean_w']:.1f} "
            f"| {row['gross_rows_per_joule']:.0f} |"
        )

    print("Saved ->", path)


if __name__ == "__main__":
    main()
