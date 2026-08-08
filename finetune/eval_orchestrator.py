"""Three-way DAG-planning accuracy: 0.8B-base vs 0.8B-finetuned vs big API model.

Metric on the held-out eval set (finetune/data/eval.jsonl):
  - DAG accuracy: exact match of the op+dependency signature vs gold.
  - Executable rate: output parses to a valid DAG using only known ops.
This produces the headline "small model matches big model" comparison table.

Run in the training env:
  python finetune/eval_orchestrator.py
Writes finetune/outputs/eval_orchestrator.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.infer import LocalOrchestrator, parse_dag  # noqa: E402
from finetune.orchestrator_prompt import SYSTEM, build_user, canonical_signature  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

enable_utf8()
HERE = Path(__file__).resolve().parent
EVAL = HERE / "data" / "eval.jsonl"
ADAPTER = HERE / "outputs" / "qwen3p5-0p8b-orchestrator"
_VALID_OPS = {"text_clean", "chunker", "medical_ner", "medical_re", "triple_validator"}


def load_eval() -> list[dict]:
    rows = []
    for line in open(EVAL, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        goal = next(m["content"] for m in msgs if m["role"] == "user")
        gold = json.loads(next(m["content"] for m in msgs if m["role"] == "assistant"))["dag"]
        rows.append({"goal": goal, "gold": gold})
    return rows


def is_executable(dag: list[dict]) -> bool:
    return bool(dag) and all(isinstance(n, dict) and n.get("op") in _VALID_OPS for n in dag)


def score(name: str, planner, rows: list[dict]) -> dict:
    exact = execu = 0
    for r in rows:
        dag = planner(r["goal"])
        if is_executable(dag):
            execu += 1
        if canonical_signature(dag) == canonical_signature(r["gold"]):
            exact += 1
    n = len(rows)
    res = {"system": name, "dag_accuracy": round(exact / n, 3), "executable_rate": round(execu / n, 3), "n": n}
    print(f"  {name:24s} DAG acc={res['dag_accuracy']:.1%}  executable={res['executable_rate']:.1%}")
    return res


def main() -> None:
    rows = load_eval()
    print(f"Eval set: {len(rows)} held-out goals\n")
    from modelscope import snapshot_download
    base = snapshot_download("Qwen/Qwen3.5-0.8B")

    results = []

    # 1) 0.8B base (zero-shot, no adapter)
    base_orch = LocalOrchestrator(base, adapter_path=None)
    results.append(score("Qwen3.5-0.8B (base)", base_orch.plan, rows))
    del base_orch

    # 2) 0.8B fine-tuned
    if ADAPTER.exists():
        ft_orch = LocalOrchestrator(base, adapter_path=str(ADAPTER))
        results.append(score("Qwen3.5-0.8B (LoRA)", ft_orch.plan, rows))
        del ft_orch
    else:
        print("  (no adapter found; run train_lora.py first)")

    # 3) big API model (upper-bound reference)
    try:
        from medigraph.llm.client import LLMClient
        llm = LLMClient()

        def big_plan(goal: str):
            return parse_dag(llm.chat(build_user(goal), system=SYSTEM, temperature=0.0))

        results.append(score(f"big API ({llm.config.model})", big_plan, rows))
    except Exception as exc:  # noqa: BLE001
        print(f"  (big-model eval skipped: {exc})")

    report = {"eval_size": len(rows), "results": results}
    (HERE / "outputs").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs" / "eval_orchestrator.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n================ DAG PLANNING ACCURACY ================")
    for r in results:
        print(f"  {r['system']:28s} acc={r['dag_accuracy']:.1%}  exec={r['executable_rate']:.1%}")
    print(f"  (saved -> {HERE / 'outputs' / 'eval_orchestrator.json'})")


if __name__ == "__main__":
    main()
