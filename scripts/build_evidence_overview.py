r"""Build a single navigation page linking the scattered demo/BI artifacts.

Not a new dashboard -- every number and chart here already exists in
`outputs/*.json`/`outputs/*.html`; this script only reads them and lays out
links + a headline metrics table + one real DAG lineage trace, so a reviewer
has one entry point instead of hunting through the outputs/ directory.

    python scripts/build_evidence_overview.py
"""
from __future__ import annotations

import html as html_lib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

OUT = Path(OUTPUTS_DIR) / "evidence_overview.html"


def _load(name: str) -> dict | None:
    path = Path(OUTPUTS_DIR) / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _get(data: dict | None, path: str, default="—"):
    if data is None:
        return default
    current = data
    for part in path.replace("]", "").split("."):
        if "[" in part:
            key, index = part.split("[")
            current = current[key] if key else current
            try:
                current = current[int(index)]
            except (IndexError, TypeError):
                return default
        else:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
    return current


def metrics_table() -> str:
    v1 = _load("eval_neural_cmeie_v1_dev.json")
    v2 = _load("eval_neural_cmeie_dev.json")
    linking = _load("eval_entity_linking.json")
    kg_scale = _load("kg_scale_report.json")
    nl2sql_hard = _load("eval_nl2sql_nl2sql_hard_natural.json")
    orchestrator = _load("../finetune/outputs/eval_orchestrator.json")
    if orchestrator is None:
        orchestrator = _load("eval_orchestrator.json")
    npu = None
    npu_path = Path(OUTPUTS_DIR).parent / "NPU" / "NPU_results" / "summary.json"
    if npu_path.exists():
        npu = json.loads(npu_path.read_text(encoding="utf-8"))
    ragas = _load("eval_ragas_kg_qa.json")

    rows = [
        ("CMeIE-V1 严格 SPO-F1", f"{_get(v1, 'end_to_end_triple_micro_strict.f1'):.4f}"
         if isinstance(_get(v1, "end_to_end_triple_micro_strict.f1"), float) else "—",
         "outputs/eval_neural_cmeie_v1_dev.json"),
        ("CMeIE-V2 dev 实体 F1", f"{_get(v2, 'entity_micro.f1'):.4f}"
         if isinstance(_get(v2, "entity_micro.f1"), float) else "—",
         "outputs/eval_neural_cmeie_dev.json"),
        ("实体链接 in-KB / NIL 拒绝",
         f"{_get(linking, 'in_kb_linking_accuracy', '—')} / {_get(linking, 'nil_rejection_rate', '—')}",
         "outputs/eval_entity_linking.json"),
        ("自产知识图谱规模",
         f"{_get(kg_scale, 'graph.num_entities', '—')} 实体 / {_get(kg_scale, 'graph.num_triples', '—')} 三元组",
         "outputs/kg_scale_report.json"),
        ("NL2SQL 非模板集（双库执行等价）",
         f"{_get(nl2sql_hard, 'dual_database_execution_accuracy', '—')}"
         f"（{_get(nl2sql_hard, 'samples', '—')} 题）",
         "outputs/eval_nl2sql_nl2sql_hard_natural.json"),
        ("0.8B LoRA 编排 DAG 准确率",
         f"{_get(orchestrator, 'results[1].dag_accuracy', '—')}",
         "finetune/outputs/eval_orchestrator.json"),
        ("NPU 融合算子吞吐 / 能效",
         f"{_get(npu, 'fused_compare_repeated.speedup.throughput', 0):.2f}× / "
         f"{_get(npu, 'fused_energy.speedup.gross_energy_efficiency', 0):.2f}×"
         if npu else "—",
         "NPU/NPU_results/summary.json"),
    ]
    if ragas is not None:
        agg = ragas.get("ragas_aggregate", {})
        rows.append((
            "GraphRAG 真实回答（Ragas）",
            " / ".join(f"{k}={v}" for k, v in agg.items() if v is not None),
            "outputs/eval_ragas_kg_qa.json",
        ))

    body = "".join(
        f"<tr><td>{html_lib.escape(label)}</td><td class='num'>{html_lib.escape(str(value))}</td>"
        f"<td><code>{html_lib.escape(src)}</code></td></tr>"
        for label, value, src in rows
    )
    return f"<table><tr><th>指标</th><th>结果</th><th>证据文件</th></tr>{body}</table>"


def lineage_table() -> str:
    report = _load("task1_report.json")
    if not report or not report.get("lineage"):
        return "<p>（尚未生成 outputs/task1_report.json；运行 demos/demo_task1_dataproc.py 后重新生成本页）</p>"
    rows = "".join(
        f"<tr><td>{html_lib.escape(item['node'])}</td><td>{html_lib.escape(item['op'])}</td>"
        f"<td class='status-{item['status']}'>{html_lib.escape(item['status'])}</td>"
        f"<td class='num'>{item.get('seconds', 0)}s</td></tr>"
        for item in report["lineage"]
    )
    summary = report.get("report", {})
    return (
        f"<p>目标：{html_lib.escape(report.get('goal',''))}　"
        f"（{summary.get('nodes_success',0)}/{summary.get('nodes_total',0)} 节点成功，"
        f"{summary.get('total_seconds',0)}s，产物 {summary.get('produced_valid_triples',0)} 条有效三元组）</p>"
        f"<table><tr><th>节点</th><th>算子</th><th>状态</th><th>耗时</th></tr>{rows}</table>"
    )


CHARTS = [
    ("outputs/task3_complete_bar.html", "任务三 · 各疾病就诊人次（柱状图）"),
    ("outputs/task3_complete_line.html", "任务三 · 月度就诊量趋势（折线图）"),
    ("outputs/task3_complete_pie.html", "任务三 · 男女就诊比例（饼图）"),
    ("outputs/task3_complete_graph.html", "任务三 · 高血压并发症/用药关系（图谱）"),
    ("outputs/task3_closed_loop.html", "任务三 · 数据-知识-洞察闭环示例"),
    ("outputs/new_case_cardiometabolic_20260701.html", "端到端案例 · 心血管代谢新病例建图"),
    ("outputs/nexent_demo_graph.html", "Nexent 演示 · 知识图谱可视化"),
    ("outputs/top10_disease_has_symptom_bar.html", "Top10 疾病症状关联数（柱状图）"),
    ("outputs/top5_cost_patients.html", "Top5 高花费患者"),
]


def chart_links() -> str:
    items = []
    for relative, label in CHARTS:
        path = Path(OUTPUTS_DIR).parent / relative
        exists = path.exists()
        cls = "" if exists else " class='missing'"
        note = "" if exists else "（尚未生成，运行对应 demo 后可用）"
        items.append(f"<li{cls}><a href='../{relative}'>{html_lib.escape(label)}</a>{note}</li>")
    return "<ul class='charts'>" + "".join(items) + "</ul>"


TEMPLATE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>MediGraph Agent · 证据总览</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
h1{{color:#2c3e50}} h2{{color:#34495e;border-bottom:2px solid #eef6ff;padding-bottom:4px;margin-top:36px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:8px 10px;text-align:left}}
th{{background:#f5f7fa}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
code{{background:#f5f5f5;padding:1px 4px;border-radius:3px;font-size:.85em}}
.status-success{{color:#256029}} .status-failed{{color:#c0392b}}
ul.charts{{line-height:2}} ul.charts li.missing{{color:#999}}
.note{{background:#fff8e1;border-left:4px solid #f39c12;padding:10px 14px;margin:16px 0;font-size:.92em}}
</style></head><body>
<h1>MediGraph Agent · 证据总览</h1>
<p class="note">本页仅汇总/链接仓内已落盘的产物，不重新计算任何指标；每个数字旁标注了来源 JSON，
可直接打开核对。生成命令：<code>python scripts/build_evidence_overview.py</code>；
完整口径见 <code>docs/EVIDENCE_MAP.md</code>、<code>docs/EVALUATION_PROTOCOL.md</code>。</p>

<h2>核心实测指标</h2>
{metrics}

<h2>任务一 · DAG 执行血缘（最近一次运行）</h2>
{lineage}

<h2>BI / 图谱可视化产物</h2>
{charts}

<h2>复现命令</h2>
<pre style="background:#f5f5f5;padding:12px;overflow-x:auto">python -m pytest -q
python scripts/check_release.py
python scripts/check_claims.py --ci
python benchmarks/eval_nl2sql.py
python benchmarks/eval_nl2sql.py --gold benchmarks/nl2sql_hard_natural.json
python benchmarks/eval_ragas_kg_qa.py
python demos/demo_task3_complete.py</pre>
</body></html>"""


def main() -> None:
    enable_utf8()
    html = TEMPLATE.format(metrics=metrics_table(), lineage=lineage_table(), charts=chart_links())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
