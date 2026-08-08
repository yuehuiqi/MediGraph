"""One-command, API-free reproduction of the headline offline evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    commands = []
    if args.rebuild_dataset:
        commands.append([sys.executable, "data/prep/build_dataset.py"])
    commands.extend(
        [
            [sys.executable, "data/prep/build_fast_extractor.py"],
            [sys.executable, "benchmarks/calibrate_fast_extractor.py", "--samples", "300" if args.quick else "1200"],
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "benchmarks/eval_fast_extraction.py", "--mode", "core"],
            [
                sys.executable,
                "benchmarks/eval_fast_extraction.py",
                "--mode",
                "cmeie",
                "--split",
                "dev",
                *(["--limit", "300"] if args.quick else []),
            ],
            [sys.executable, "benchmarks/eval_entity_linking.py"],
            [sys.executable, "benchmarks/eval_kg_qa.py"],
            [sys.executable, "benchmarks/build_nl2sql_stress.py"],
            [sys.executable, "benchmarks/eval_nl2sql.py"],
            [
                sys.executable,
                "benchmarks/eval_nl2sql.py",
                "--gold",
                "benchmarks/nl2sql_stress_128.json",
            ],
            [sys.executable, "scripts/check_release.py"],
            [sys.executable, "scripts/check_claims.py", "--ci"],
        ]
    )
    records = []
    for command in commands:
        print("\n>", " ".join(command), flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, text=True)
        record = {
            "command": command,
            "returncode": completed.returncode,
            "seconds": round(time.perf_counter() - started, 3),
        }
        records.append(record)
        if completed.returncode:
            break
    artifacts = {}
    for relative in (
        "data/models/fast_extractor.json",
        "data/models/entity_linker.json",
        "data/models/temperature_calibration.json",
        "outputs/calibration_report.json",
        "outputs/eval_fast_core.json",
        "outputs/eval_fast_cmeie_dev.json",
        "outputs/eval_neural_cmeie_dev.json",
        "outputs/eval_ensemble_cmeie_dev.json",
        "outputs/eval_neural_cmeie_v1_dev.json",
        "outputs/eval_ensemble_cmeie_v1_dev.json",
        "outputs/eval_entity_linking.json",
        "outputs/eval_entity_linking_real.json",
        "outputs/eval_kg_qa.json",
        "outputs/eval_nl2sql_nl2sql_hard_natural.json",
        "outputs/CMeIE_test_pred.jsonl",
        "outputs/kg_scale_report.json",
        "outputs/eval_nl2sql.json",
        "outputs/eval_nl2sql_nl2sql_stress_128.json",
        "data/models/neural_extractor/train_log.json",
    ):
        path = ROOT / relative
        if path.exists():
            artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    report = {
        "started_at": started_at,
        "python": sys.version,
        "platform": platform.platform(),
        "quick": args.quick,
        "success": all(record["returncode"] == 0 for record in records) and len(records) == len(commands),
        "commands": records,
        "artifacts": artifacts,
    }
    output = ROOT / "outputs" / "reproduction_manifest.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest -> {output}")
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
