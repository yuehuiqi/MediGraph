"""Run the synthetic cardiometabolic acceptance case through local MCP tools.

The generated artifacts are intentionally separate from the competition graph,
so the case can be replayed in Nexent without changing benchmark outputs.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_server.server import (  # noqa: E402
    build_medical_kg,
    inspect_extraction_models,
    medical_kg_qa,
)


CASE_DIR = ROOT / "data" / "demo_cases" / "cardiometabolic_20260701"
OUTPUT_DIR = ROOT / "outputs"
GRAPH_NAME = "new_case_cardiometabolic_local_20260701.json"


def _decode(result: str) -> dict:
    return json.loads(result)


def _write(name: str, payload: dict) -> Path:
    path = OUTPUT_DIR / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    documents = []
    for path in sorted(CASE_DIR.glob("*.txt")):
        documents.append(
            {
                "file": path.name,
                "text": path.read_text(encoding="utf-8").strip(),
            }
        )
    if not documents:
        raise SystemExit(f"No case documents found under {CASE_DIR}")

    combined = "\n\n".join(
        f"【来源文件：{item['file']}】\n{item['text']}" for item in documents
    )
    build = _decode(
        build_medical_kg(
            text=combined,
            source_name="cardiometabolic_20260701_combined.txt",
            graph_name=GRAPH_NAME,
            append=False,
            max_chunks=8,
        )
    )
    qa_heart_failure = _decode(
        medical_kg_qa(
            question="心力衰竭有哪些典型症状和检查依据？请给出证据三元组与来源。",
            graph_name=GRAPH_NAME,
            hops=2,
        )
    )
    qa_hypertension = _decode(
        medical_kg_qa(
            question="高血压可能并发哪些疾病？请给出证据三元组、置信度与来源。",
            graph_name=GRAPH_NAME,
            hops=2,
        )
    )
    model_audit = _decode(inspect_extraction_models())

    build_path = _write("new_case_build_response.json", build)
    qa1_path = _write("new_case_qa_heart_failure.json", qa_heart_failure)
    qa2_path = _write("new_case_qa_hypertension.json", qa_hypertension)
    audit_path = _write("new_case_model_audit.json", model_audit)
    summary = {
        "case": "cardiometabolic_20260701",
        "generated_at": datetime.now().astimezone().isoformat(),
        "synthetic_non_patient_data": True,
        "input_files": documents,
        "graph": {
            "json": build.get("graph_file"),
            "html": build.get("graph_html"),
            "stats": build.get("graph"),
            "quality": build.get("quality"),
        },
        "questions": [
            {
                "question": qa_heart_failure.get("question"),
                "answer": qa_heart_failure.get("answer"),
                "evidence_total": qa_heart_failure.get("evidence_total"),
                "refused": qa_heart_failure.get("refused"),
            },
            {
                "question": qa_hypertension.get("question"),
                "answer": qa_hypertension.get("answer"),
                "evidence_total": qa_hypertension.get("evidence_total"),
                "refused": qa_hypertension.get("refused"),
            },
        ],
        "artifacts": [
            str(build_path),
            str(qa1_path),
            str(qa2_path),
            str(audit_path),
        ],
    }
    summary_path = _write("new_case_summary.json", summary)
    print(json.dumps({"ok": True, "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
