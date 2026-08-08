"""Synthesize the (NL goal -> operator DAG) training set with a big API model.

For each canonical pipeline pattern we ask the big model to write many diverse
Chinese/English natural-language data-processing goals that map to that exact
pipeline; each goal is paired with the pattern's trustworthy gold DAG. This is the
"use the big model + DataMate-style synthesis to train a small model" approach.

Run in the env that has `openai` + a working API key (e.g. the current one):
  python finetune/synth_data.py --per-pattern 60
Outputs finetune/data/train.jsonl and finetune/data/eval.jsonl (chat format).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.orchestrator_prompt import SYSTEM, build_user, dag_to_output, patterns  # noqa: E402
from medigraph.llm.client import LLMClient  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

enable_utf8()
DATA_DIR = Path(__file__).resolve().parent / "data"

_GEN_SYSTEM = "你是数据标注助手，只输出 JSON 数组，不要解释。"
_GEN_PROMPT = """请生成 {n} 条**多样化**的自然语言「数据处理需求」，这些需求都对应同一条处理流程：
流程意图：{intent}
对应算子序列：{ops}

要求：
1. 用词、句式尽量多样（命令式/口语/正式/含医疗场景词），覆盖中文为主、约 1/4 用英文。
2. 只描述需求，不要出现算子名(text_clean/chunker 等)或 JSON。
3. 不要编号。输出 JSON 对象，键名固定为 goals：{{"goals":["需求1","需求2",...]}}。"""


def _gen_batch(llm: LLMClient, intent: str, ops: list[str], n: int) -> list[str]:
    prompt = _GEN_PROMPT.format(n=n, intent=intent, ops=" -> ".join(ops) or "(无)")
    try:
        data = llm.chat_json(prompt, system=_GEN_SYSTEM, default=[])
    except Exception as exc:  # noqa: BLE001 - one flaky call must not kill the run
        print(f"    ! batch failed: {exc}")
        return []
    goals = data if isinstance(data, list) else data.get("goals", []) if isinstance(data, dict) else []
    return [str(g).strip() for g in goals if str(g).strip()]


def gen_goals(llm: LLMClient, intent: str, ops: list[str], n: int, batch: int = 20) -> list[str]:
    """Generate in small batches so flaky/slow API calls don't time out or lose all."""
    out: list[str] = []
    remaining = n
    while remaining > 0:
        got = _gen_batch(llm, intent, ops, min(batch, remaining))
        if not got:
            break  # give up this pattern's remaining quota, keep what we have
        out.extend(got)
        remaining -= batch
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pattern", type=int, default=60, help="goals per pattern")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()
    rng = random.Random(args.seed)
    records: list[dict] = []

    for pat in patterns():
        ops = [n["op"] for n in pat["dag"]]
        output = dag_to_output(pat["dag"])
        goals = gen_goals(llm, pat["intent"], ops, args.per_pattern)
        # de-dup goals
        seen, uniq = set(), []
        for g in goals:
            if g.lower() not in seen:
                seen.add(g.lower())
                uniq.append(g)
        print(f"  pattern '{pat['intent'][:24]}': {len(uniq)} goals")
        for g in uniq:
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user(g)},
                    {"role": "assistant", "content": output},
                ]
            })

    rng.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_frac))
    eval_rec, train_rec = records[:n_eval], records[n_eval:]

    def _write(recs, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write(train_rec, DATA_DIR / "train.jsonl")
    _write(eval_rec, DATA_DIR / "eval.jsonl")
    print(f"\nTotal {len(records)} examples -> train={len(train_rec)}, eval={len(eval_rec)}")
    print(f"  {DATA_DIR/'train.jsonl'}\n  {DATA_DIR/'eval.jsonl'}")
    print(f"  LLM stats: {llm.stats.summary()}")


if __name__ == "__main__":
    main()
