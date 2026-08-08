"""Fail-fast checks for a safe, reviewable competition submission."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "requirements-docker.txt",
    "requirements-docker-neural.txt",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    "scripts/docker_up.ps1",
    "scripts/docker_down.ps1",
    ".env.example",
    "docs/README.md",
    "docs/EVALUATION_PROTOCOL.md",
    "docs/MODEL_AND_DATA_CARDS.md",
    "docs/EVIDENCE_MAP.md",
    "data/external_manifest.json",
]
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|secret)\s*[:=]\s*[\"']?(?!x{4,})[A-Za-z0-9_-]{20,}"),
]
SCAN_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".ps1", ".sh"}
SKIP_PARTS = {".git", "__pycache__", ".venv", ".venv-neural", "venv", "finetune", "outputs", "演示视频录制与截图"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="skip local artifact requirements")
    args = parser.parse_args()
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if (ROOT / ".env").exists():
        print("INFO: local .env exists and is excluded by .gitignore")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            matches = list(pattern.finditer(text))
            real_matches = [
                match
                for match in matches
                if not (
                    match.group(0).lower().startswith("sk-")
                    and set(match.group(0)[3:].lower()) <= {"x"}
                )
            ]
            if real_matches:
                errors.append(f"possible secret in {path.relative_to(ROOT)}")
                break
    # Trained weights are not part of this GitHub distribution -- see the README's
    # "关于本仓库" section for what was left out and how to regenerate it. Only the
    # artifacts that actually ship here are required, so a fresh clone passes.
    if not args.ci:
        for relative in (
            "data/models/fast_extractor.json",
            "data/models/entity_linker.json",
            "outputs/calibration_report.json",
            "outputs/eval_fast_core.json",
            "outputs/eval_fast_cmeie_dev.json",
            "outputs/eval_neural_cmeie_dev.json",
            "outputs/eval_ensemble_cmeie_dev.json",
            "outputs/eval_neural_cmeie_v1_dev.json",
            "outputs/eval_ensemble_cmeie_v1_dev.json",
            "outputs/eval_entity_linking.json",
            "outputs/eval_kg_qa.json",
            "outputs/eval_nl2sql.json",
            "outputs/eval_nl2sql_nl2sql_stress_128.json",
            "outputs/kg_scale_report.json",
        ):
            if not (ROOT / relative).is_file():
                errors.append(f"missing reproducibility artifact: {relative}")
    result = {"passed": not errors, "errors": errors, "required_files": len(REQUIRED)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
