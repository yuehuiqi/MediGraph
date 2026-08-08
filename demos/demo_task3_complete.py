"""One-command Task 3 demo: SQL/GRAPH planning, BI reports and evidence audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from mcp_server.server import analyze_medical_data, inspect_analysis_assets  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

enable_utf8()

_CASES = [
    ("各疾病的就诊人次是多少", "task3_complete_bar.html"),
    ("2024年每个月的就诊量趋势", "task3_complete_line.html"),
    ("男性和女性的就诊比例", "task3_complete_pie.html"),
    ("高血压有哪些并发症和推荐用药关系", "task3_complete_graph.html"),
]


"""Graph consumed by this demo.

Task 3 must demonstrably reuse the *self-produced* Task-2 output, so this points at
`graph_scaled.json` (43,566 entities / 70,790 triples, neural GPLinker extraction
over 25,492 documents) rather than the imported CM3KG baseline `graph.json`.
Falls back to the baseline when the scaled graph has not been built yet.
"""
_GRAPH_NAME = (
    "graph_scaled.json"
    if (OUTPUTS_DIR / "graph_scaled.json").exists()
    else "graph.json"
)


def main() -> None:
    analyses = []
    print(f"[task3] graph source: {_GRAPH_NAME}")
    for index, (question, report_name) in enumerate(_CASES, start=1):
        print(f"\n=== Task 3 case {index}: {question} ===")
        result = json.loads(
            analyze_medical_data(
                question,
                graph_name=_GRAPH_NAME,
                report_name=report_name,
                n_visits=600,
                seed=42,
            )
        )
        analyses.append(result)
        print(
            json.dumps(
                {
                    "plan": result.get("plan"),
                    "chart_type": result.get("chart_type"),
                    "row_count": result.get("row_count"),
                    "sql": result.get("sql"),
                    "citations": result.get("citations"),
                    "insight": result.get("insight"),
                    "report_html": result.get("report_html"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    audit = json.loads(inspect_analysis_assets(_GRAPH_NAME))
    report = {
        "task": "Task 3 - graph-aware data analysis and BI visualization",
        "graph_source": _GRAPH_NAME,
        "analyses": analyses,
        "audit": audit,
    }
    report_path = write_json(report, OUTPUTS_DIR / "task3_complete_report.json")
    print(f"\nComplete report -> {report_path}")


if __name__ == "__main__":
    main()
