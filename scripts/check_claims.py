r"""Cross-check numeric claims in README.md / docs/EVIDENCE_MAP.md against the
evaluation JSON that is supposed to back them.

Why this exists
----------------
A number typed into a Markdown table has no relationship to the JSON file next
to it unless something enforces one. This project has already shipped that
exact failure twice, independently discovered while doing this work: README
claimed CMeIE-V1 strict SPO-F1 = 0.6146 while EVIDENCE_MAP claimed 0.6156 for
the *same* metric (only the former matched `outputs/eval_neural_cmeie_v1_dev.json`);
and README's "101 passed" test count silently drifted from the real suite
size as tests were added. Both are the same bug: a hand-typed number nobody
re-derives.

This script cannot read prose and understand what a number means -- it checks
an explicit, curated table below (`CLAIMS`) that pairs each headline number
with the exact JSON file/path that must produce it and the doc line it must
appear on. Anyone changing a reported metric must update `CLAIMS` in the same
change, which is the point: a stale number now fails CI instead of shipping.

Usage
    python scripts/check_claims.py            # human-readable report
    python scripts/check_claims.py --ci        # same, exits non-zero on any mismatch
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Field:
    """One number to verify: `json_path` into `json_file`, formatted, must
    appear verbatim on a line of `doc_file` containing `label`."""

    json_path: str  # dotted/bracket path, e.g. "graph.num_entities" or "results[1].dag_accuracy"
    fmt: str = "{}"  # format spec applied to the extracted value


@dataclass
class Claim:
    doc_file: str
    label: str  # substring identifying the line(s) in doc_file to check
    json_file: str
    fields: list[Field] = field(default_factory=list)
    note: str = ""


def _extract(data: object, path: str) -> object:
    """Walk a dotted/bracketed path like `graph.num_entities` or `results[1].f1`."""
    current = data
    for part in path.replace("]", "").split("."):
        if "[" in part:
            key, index = part.split("[")
            if key:
                current = current[key]
            current = current[int(index)]
        else:
            current = current[part]
    return current


# --------------------------------------------------------------------------- #
# claims table -- one entry per headline number in README.md / EVIDENCE_MAP.md
# --------------------------------------------------------------------------- #
CLAIMS: list[Claim] = [
    # -- CMeIE-V1 strict SPO-F1: the exact drift this script exists to catch -- #
    Claim(
        "README.md", "CMeIE-V1 · 严格 SPO-F1",
        "outputs/eval_neural_cmeie_v1_dev.json",
        [Field("end_to_end_triple_micro_strict.f1", "{:.4f}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "CMeIE-V1 · 严格 SPO-F1",
        "outputs/eval_neural_cmeie_v1_dev.json",
        [Field("end_to_end_triple_micro_strict.f1", "{:.4f}")],
    ),
    Claim(
        # The row-2 sighting: same number restated inline in the Task-2 scoring
        # table, distinct from the row-1 "本作品实测硬指标汇总" table above --
        # this is the exact line that was still stale after the first fix.
        "docs/EVIDENCE_MAP.md", "CMeIE-V1 严格 SPO-F1",
        "outputs/eval_neural_cmeie_v1_dev.json",
        [Field("end_to_end_triple_micro_strict.f1", "{:.4f}")],
    ),
    # -- CMeIE-V2 dev: entity F1 / end-to-end triple F1 (+ ensemble) -- #
    Claim(
        "README.md", "CMeIE-V2 dev 实体 F1",
        "outputs/eval_neural_cmeie_dev.json",
        [Field("entity_micro.f1", "{:.4f}"), Field("end_to_end_triple_micro.f1", "{:.4f}")],
    ),
    Claim(
        "README.md", "CMeIE-V2 dev 实体 F1",
        "outputs/eval_ensemble_cmeie_dev.json",
        [Field("triple_micro.conf_union.f1", "{:.4f}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "L1 神经抽取 · 实体 F1",
        "outputs/eval_neural_cmeie_dev.json",
        [Field("entity_micro.f1", "{:.4f}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "L1 神经抽取 · 端到端三元组 F1",
        "outputs/eval_neural_cmeie_dev.json",
        [Field("end_to_end_triple_micro.f1", "{:.4f}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "L1 神经抽取 · 端到端三元组 F1",
        "outputs/eval_ensemble_cmeie_dev.json",
        [Field("triple_micro.conf_union.f1", "{:.4f}")],
    ),
    # -- entity linking -- #
    Claim(
        "README.md", "实体链接 in-KB 准确率",
        "outputs/eval_entity_linking.json",
        [Field("in_kb_linking_accuracy", "{:.4f}"), Field("nil_rejection_rate", "{:.3f}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "实体链接 in-KB 准确率",
        "outputs/eval_entity_linking.json",
        [Field("in_kb_linking_accuracy", "{:.4f}"), Field("nil_rejection_rate", "{:.3f}")],
    ),
    # -- self-produced graph scale -- #
    Claim(
        "README.md", "自产知识图谱规模",
        "outputs/kg_scale_report.json",
        [Field("graph.num_entities", "{:,}"), Field("graph.num_triples", "{:,}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "自产知识图谱规模",
        "outputs/kg_scale_report.json",
        [Field("graph.num_entities", "{:,}"), Field("graph.num_triples", "{:,}")],
    ),
    # -- calibration -- #
    Claim(
        "docs/EVIDENCE_MAP.md", "标定后预测级 ECE",
        "outputs/calibration_report.json",
        [Field("ece_after", "{:.3f}"), Field("ece_before", "{:.3f}")],
    ),
    # -- 0.8B LoRA orchestrator -- #
    Claim(
        "README.md", "0.8B LoRA 编排 DAG 准确率",
        "finetune/outputs/eval_orchestrator.json",
        [Field("results[1].dag_accuracy", "{:.3f}"), Field("results[1].executable_rate", "{:.3f}")],
        note="results[1] must stay the LoRA row; results[0]=base, results[2]=big API.",
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "0.8B LoRA 编排 DAG 准确率",
        "finetune/outputs/eval_orchestrator.json",
        [Field("results[1].dag_accuracy", "{:.3f}"), Field("results[1].executable_rate", "{:.3f}")],
    ),
    # -- NPU -- #
    Claim(
        "README.md", "NPU 融合算子吞吐",
        "NPU/NPU_results/summary.json",
        [Field("fused_compare_repeated.speedup.throughput", "{:.2f}")],
    ),
    Claim(
        "README.md", "整卡能效",
        "NPU/NPU_results/summary.json",
        [Field("fused_energy.speedup.gross_energy_efficiency", "{:.2f}")],
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "NPU 融合算子：吞吐",
        "NPU/NPU_results/summary.json",
        [
            Field("fused_compare_repeated.speedup.throughput", "{:.2f}"),
            Field("fused_energy.speedup.gross_energy_efficiency", "{:.2f}"),
        ],
    ),
    # -- NL2SQL non-template set: the number that was stale (85.7%) when this
    # script was first written, and the reason it exists. -- #
    Claim(
        "README.md", "NL2SQL：人工 16 题",
        "outputs/eval_nl2sql_nl2sql_hard_natural.json",
        [Field("dual_database_execution_accuracy", "{:.0%}"), Field("samples", "{}")],
        note="Sample count is asserted too so a shrunk set is caught, not just a rounded-up percentage.",
    ),
    Claim(
        "docs/EVIDENCE_MAP.md", "NL2SQL：人工 16 / 压力集 128",
        "outputs/eval_nl2sql_nl2sql_hard_natural.json",
        [Field("dual_database_execution_accuracy", "{:.0%}"), Field("samples", "{}")],
    ),
    # -- NL2SQL held-out probe: the counterweight to the three 100%s. If the
    # router regresses on questions it was never tuned against, the honest
    # caveat in EVALUATION_PROTOCOL.md must move with it. -- #
    Claim(
        "docs/EVALUATION_PROTOCOL.md", "修复后留出集",
        "outputs/eval_nl2sql_holdout.json",
        [Field("correct", "{}"), Field("router_answered", "{}")],
        note="Asserts the '25/25' in the honest-caveat section still matches the artifact.",
    ),
]


# --------------------------------------------------------------------------- #
def check(claim: Claim) -> list[str]:
    """Return a list of problem descriptions for this claim (empty = OK)."""
    problems: list[str] = []
    doc_path = ROOT / claim.doc_file
    json_path = ROOT / claim.json_file
    if not doc_path.exists():
        return [f"doc file missing: {claim.doc_file}"]
    if not json_path.exists():
        return [f"json file missing: {claim.json_file} (run the eval script that produces it)"]

    lines = [line for line in doc_path.read_text(encoding="utf-8").splitlines() if claim.label in line]
    if not lines:
        return [f"label {claim.label!r} not found anywhere in {claim.doc_file}"]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for f in claim.fields:
        try:
            value = _extract(data, f.json_path)
        except (KeyError, IndexError, TypeError) as exc:
            problems.append(f"json_path {f.json_path!r} not found in {claim.json_file} ({exc})")
            continue
        expected = f.fmt.format(value)
        if not any(expected in line for line in lines):
            problems.append(
                f"expected {expected!r} (from {claim.json_file}:{f.json_path}) "
                f"not found on any {claim.label!r} line of {claim.doc_file}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="exit 1 on any mismatch")
    args = parser.parse_args()

    ok = 0
    failures: list[tuple[Claim, list[str]]] = []
    for claim in CLAIMS:
        problems = check(claim)
        if problems:
            failures.append((claim, problems))
        else:
            ok += 1

    print(f"claims checked: {len(CLAIMS)}  ok: {ok}  mismatched: {len(failures)}")
    for claim, problems in failures:
        print(f"\n[MISMATCH] {claim.doc_file} :: {claim.label!r}  <-  {claim.json_file}")
        for problem in problems:
            print(f"    - {problem}")
        if claim.note:
            print(f"    note: {claim.note}")

    if failures:
        print(
            "\nA doc number does not match the JSON that is supposed to back it. "
            "Either the doc is stale (update it) or the CLAIMS entry is wrong "
            "(fix scripts/check_claims.py)."
        )
        return 1 if args.ci else 0
    print("all claimed numbers match their JSON source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
