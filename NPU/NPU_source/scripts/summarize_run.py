from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
mode = sys.argv[2] if len(sys.argv) > 2 else "full"


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def system_row(report, name):
    if not isinstance(report, dict):
        return None

    results = report.get("results", report)

    if isinstance(results, dict):
        return results.get(name)

    if isinstance(results, list):
        for row in results:
            if (
                isinstance(row, dict)
                and row.get("system") == name
            ):
                return row

    return None


def pick(row, *names):
    if not isinstance(row, dict):
        return None
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def cv(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return (
        statistics.stdev(values)
        / statistics.mean(values)
        * 100.0
    )


def aggregate_runs(paths, systems):
    reports = [
        load_json(path)
        for path in sorted(paths)
    ]
    reports = [item for item in reports if item]

    output = {}

    for system in systems:
        rows = [
            system_row(report, system)
            for report in reports
        ]
        rows = [row for row in rows if row]

        throughputs = [
            pick(
                row,
                "throughput_rows_per_s",
                "throughput",
                "rows_per_s",
            )
            for row in rows
        ]
        p50s = [
            pick(row, "latency_p50_ms", "p50_ms")
            for row in rows
        ]
        p99s = [
            pick(row, "latency_p99_ms", "p99_ms")
            for row in rows
        ]
        errors = [
            pick(
                row,
                "max_abs_diff_vs_cpu",
                "max_abs_diff",
            )
            for row in rows
        ]
        errors = [value for value in errors if value is not None]

        output[system] = {
            "runs": len(rows),
            "throughput_median_rows_per_s":
                median(throughputs),
            "throughput_cv_percent":
                cv(throughputs),
            "p50_median_ms": median(p50s),
            "p99_median_ms": median(p99s),
            "max_abs_diff_worst":
                max(errors) if errors else None,
        }

    return output


def aggregate_e2e(paths):
    reports = [
        load_json(path)
        for path in sorted(paths)
    ]
    reports = [item for item in reports if item]

    output = {}

    for system in ("npu_torch", "npu_ascendc"):
        rows = [
            system_row(report, system)
            for report in reports
        ]
        rows = [row for row in rows if row]

        output[system] = {
            "runs": len(rows),
            "mean_median_ms": median([
                pick(row, "mean_ms")
                for row in rows
            ]),
            "p50_median_ms": median([
                pick(row, "p50_ms")
                for row in rows
            ]),
            "p99_median_ms": median([
                pick(row, "p99_ms")
                for row in rows
            ]),
        }

    return output


def profile_task(directory):
    directory = Path(directory)
    paths = list(directory.rglob("OpBasicInfo.csv"))
    if not paths:
        return None

    path = max(paths, key=lambda item: item.stat().st_mtime)

    with path.open(
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        row = next(csv.DictReader(file), None)

    if not row:
        return None

    return {
        "op_name": row.get("Op Name"),
        "op_type": row.get("Op Type"),
        "task_duration_us": float(
            row["Task Duration(us)"]
        ),
        "block_dim": int(row["Block Dim"]),
        "source": str(path.relative_to(run_dir)),
    }


pure = aggregate_runs(
    (run_dir / "repeats/pure").glob("run_*.json"),
    ("cpu_torch", "npu_torch", "npu_ascendc"),
)

e2e = aggregate_e2e(
    (run_dir / "repeats/e2e").glob("run_*.json")
)

fused_compare = load_json(
    run_dir / "fused_compare_repeated.json"
)
pure_energy = load_json(
    run_dir / "energy_compare.json"
)
fused_energy = load_json(
    run_dir / "fused_energy_compare.json"
)

profiles = {
    "pure_softmax": profile_task(
        run_dir / "profiler/pure"
    ),
    "fused_softmax": profile_task(
        run_dir / "profiler/fused"
    ),
}

summary = {
    "run_id": run_dir.name,
    "mode": mode,
    "pure_softmax_repeated": pure,
    "end_to_end_repeated": e2e,
    "pure_energy": pure_energy,
    "fused_compare_repeated": fused_compare,
    "fused_energy": fused_energy,
    "profiler": profiles,
}

validation_errors = []

torch_throughput = pure.get(
    "npu_torch", {}
).get("throughput_median_rows_per_s")
custom_throughput = pure.get(
    "npu_ascendc", {}
).get("throughput_median_rows_per_s")

if not torch_throughput or not custom_throughput:
    validation_errors.append(
        "missing pure NPU throughput"
    )
elif custom_throughput <= torch_throughput:
    validation_errors.append(
        "custom pure softmax is not faster than torch_npu"
    )

pure_error = pure.get(
    "npu_ascendc", {}
).get("max_abs_diff_worst")

if pure_error is not None and pure_error > 1e-5:
    validation_errors.append(
        f"pure correctness failed: {pure_error}"
    )

if not fused_compare:
    validation_errors.append(
        "missing fused repeated comparison"
    )
else:
    speedup = fused_compare.get(
        "speedup", {}
    ).get("throughput")

    if not speedup or speedup <= 1.0:
        validation_errors.append(
            "fused throughput speedup <= 1"
        )

    correctness = fused_compare.get(
        "correctness", {}
    )

    if correctness.get("max_abs_diff", 1) > 1e-5:
        validation_errors.append(
            "fused correctness failed"
        )

    if correctness.get(
        "masked_probability_max", 1
    ) > 1e-6:
        validation_errors.append(
            "fused mask correctness failed"
        )

if mode == "full":
    if not pure_energy:
        validation_errors.append(
            "missing pure energy report"
        )

    if not fused_energy:
        validation_errors.append(
            "missing fused energy report"
        )
    else:
        gross = fused_energy.get(
            "speedup", {}
        ).get("gross_energy_efficiency")

        if not gross or gross <= 1.0:
            validation_errors.append(
                "fused gross energy efficiency <= 1"
            )

    if not profiles["pure_softmax"]:
        validation_errors.append(
            "missing pure profiler"
        )

    if not profiles["fused_softmax"]:
        validation_errors.append(
            "missing fused profiler"
        )

summary["validation"] = {
    "passed": not validation_errors,
    "errors": validation_errors,
}

(run_dir / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)


def number(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def throughput(value):
    if value is None:
        return "-"
    return f"{value / 1e9:.3f}B"


lines = [
    "# NPU Full Benchmark Report",
    "",
    f"- Run ID: `{run_dir.name}`",
    f"- Mode: `{mode}`",
    "- Shape: `[200000, 32]`",
    "",
    "## Pure Softmax: repeated-run median",
    "",
    "| system | throughput | CV | P50 | P99 |",
    "|---|---:|---:|---:|---:|",
]

for system in (
    "cpu_torch",
    "npu_torch",
    "npu_ascendc",
):
    row = pure.get(system, {})
    lines.append(
        f"| {system} "
        f"| {throughput(row.get('throughput_median_rows_per_s'))} "
        f"| {number(row.get('throughput_cv_percent'), 2)}% "
        f"| {number(row.get('p50_median_ms'))} ms "
        f"| {number(row.get('p99_median_ms'))} ms |"
    )

lines.extend([
    "",
    "## End-to-end: repeated-run median",
    "",
    "| system | mean | P50 | P99 |",
    "|---|---:|---:|---:|",
])

for system in ("npu_torch", "npu_ascendc"):
    row = e2e.get(system, {})
    lines.append(
        f"| {system} "
        f"| {number(row.get('mean_median_ms'), 4)} ms "
        f"| {number(row.get('p50_median_ms'), 4)} ms "
        f"| {number(row.get('p99_median_ms'), 4)} ms |"
    )

if fused_compare:
    results = fused_compare.get("results", {})
    lines.extend([
        "",
        "## Fused calibrated masked Softmax",
        "",
        "| system | throughput | CV | P50 | P99 |",
        "|---|---:|---:|---:|---:|",
    ])

    for system in (
        "npu_torch_composed",
        "npu_ascendc_fused",
    ):
        row = results.get(system, {})
        lines.append(
            f"| {system} "
            f"| {throughput(row.get('throughput_median_rows_per_s'))} "
            f"| {number(row.get('throughput_cv_percent'), 2)}% "
            f"| {number(row.get('p50_median_ms'))} ms "
            f"| {number(row.get('p99_median_ms'))} ms |"
        )

    lines.extend([
        "",
        "Speedups:",
        "",
        f"- Throughput: {number(fused_compare['speedup'].get('throughput'), 2)}x",
        f"- P50: {number(fused_compare['speedup'].get('p50'), 2)}x",
        f"- P99: {number(fused_compare['speedup'].get('p99'), 2)}x",
    ])

if fused_energy:
    speedup = fused_energy.get("speedup", {})
    lines.extend([
        "",
        "## Fused energy efficiency",
        "",
        f"- Throughput speedup: {number(speedup.get('throughput'), 2)}x",
        f"- Gross rows/J improvement: {number(speedup.get('gross_energy_efficiency'), 2)}x",
        f"- Idle-subtracted rows/J improvement: {number(speedup.get('dynamic_energy_efficiency'), 2)}x",
    ])

lines.extend([
    "",
    "## Profiler",
    "",
    "| operator | task duration | blockDim |",
    "|---|---:|---:|",
])

for name, row in profiles.items():
    row = row or {}
    lines.append(
        f"| {name} "
        f"| {number(row.get('task_duration_us'), 3)} us "
        f"| {row.get('block_dim', '-')} |"
    )

lines.extend([
    "",
    "## Validation",
    "",
    "PASS" if not validation_errors else "FAIL",
])

for error in validation_errors:
    lines.append(f"- {error}")

(run_dir / "REPORT.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))

if validation_errors:
    raise SystemExit(3)
