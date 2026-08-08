"""Task 3 demo: graph-driven data analysis + BI visualization.

Builds the analytics SQLite DB from the Task-2 knowledge graph (or an embedded
example graph if Task 2 hasn't run yet), then answers natural-language analysis
questions. For each question the agent routes to SQL (statistics/trend) or GRAPH
(association), produces a natural-language insight, and writes an ECharts HTML
report (BI chart or relationship network).

Usage:
  python demos/demo_task3_analysis.py
  python demos/demo_task3_analysis.py --question "每个科室的就诊量是多少"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.analysis.analysis_agent import AnalysisAgent  # noqa: E402
from medigraph.analysis.graph_profile import load_graph  # noqa: E402
from medigraph.analysis.relational import build_db  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

enable_utf8()

_SAMPLE_QUESTIONS = [
    "各疾病的就诊人次是多少",            # SQL - statistics (bar)
    "2024年每个月的就诊量趋势",          # SQL - trend (line)
    "男性和女性的就诊比例",              # SQL - ratio (pie)
    "高血压有哪些并发症和推荐用药",      # GRAPH - association (network)
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Task3 analysis + visualization demo")
    ap.add_argument("--question", default=None)
    ap.add_argument("--n-visits", type=int, default=600)
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUTS_DIR / "analytics.db"
    graph_json = OUTPUTS_DIR / "graph.json"

    # Build the relational DB from the Task-2 graph (reuse) or an example graph.
    store, used_example = load_graph(graph_json if graph_json.exists() else None)
    summary = build_db(db_path, store, n_visits=args.n_visits, seed=42)
    src = "embedded example graph (run Task 2 to use the real one)" if used_example else "Task-2 graph.json"
    print(f"Analytics DB built from {src}: {summary['n_visits']} visits over {summary['diseases']}\n")

    agent = AnalysisAgent(str(db_path), graph_json=str(graph_json) if graph_json.exists() else None)

    questions = [args.question] if args.question else _SAMPLE_QUESTIONS
    for i, q in enumerate(questions, 1):
        print(f"\n================ Q{i}: {q} ================")
        out_html = OUTPUTS_DIR / f"task3_report_{i}.html"
        res = agent.analyze(q, out_html=out_html)
        print(f"route={res['route']}  chart={res['chart_type']}  rows={len(res['rows'])}")
        if res["sql"]:
            print(f"SQL: {res['sql']}")
        print(f"洞察: {res['insight']}")
        print(f"报告: {res['html']}  (用浏览器打开)")


if __name__ == "__main__":
    main()
