"""One-command Task 2 demo: build a traceable medical KG and answer over it.

Usage:
  python demos/demo_task2_complete.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from mcp_server.server import build_medical_kg, inspect_medical_kg, medical_kg_qa  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

enable_utf8()

_DEFAULT_TEXT = (
    "2型糖尿病患者常有多饮、多尿，建议检查糖化血红蛋白，"
    "治疗可使用二甲双胍，并可能并发糖尿病肾病。"
)
_DEFAULT_QUESTION = "2型糖尿病建议做什么检查，使用什么药物？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete Task 2 KG + GraphRAG demo")
    parser.add_argument("--text", default=_DEFAULT_TEXT)
    parser.add_argument("--question", default=_DEFAULT_QUESTION)
    parser.add_argument("--graph-name", default="nexent_demo_graph.json")
    args = parser.parse_args()

    print("=== 1) Automatic operator orchestration and KG construction ===")
    build = json.loads(
        build_medical_kg(
            args.text,
            source_name="nexent_demo_diabetes.txt",
            graph_name=args.graph_name,
            append=False,
            max_chunks=2,
        )
    )
    print(json.dumps(build, ensure_ascii=False, indent=2))

    print("\n=== 2) Intent-aware, traceable GraphRAG QA ===")
    qa = json.loads(medical_kg_qa(args.question, graph_name=args.graph_name, hops=1))
    print(json.dumps(qa, ensure_ascii=False, indent=2))

    print("\n=== 3) Ontology, graph and evaluation evidence ===")
    audit = json.loads(inspect_medical_kg(args.graph_name, sample_limit=8))
    report = {
        "demo": "medical knowledge graph generation and QA",
        "build": build,
        "qa": qa,
        "audit": audit,
    }
    report_path = write_json(report, OUTPUTS_DIR / "task2_complete_report.json")
    print(f"\nComplete report -> {report_path}")
    print(f"Interactive graph -> {build.get('graph_html')}")


if __name__ == "__main__":
    main()
