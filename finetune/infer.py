"""Load the fine-tuned 0.8B orchestrator and generate operator DAGs locally.

Provides LocalOrchestrator used by eval and (optionally) by DataProcAgent as a
local, cheap planner instead of the big API model.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.orchestrator_prompt import SYSTEM, build_user  # noqa: E402

DEFAULT_ADAPTER = Path(__file__).resolve().parent / "outputs" / "qwen3p5-0p8b-orchestrator"


class LocalOrchestrator:
    def __init__(self, base_path: str, adapter_path: str | None = None, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None, trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, goal: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_user(goal)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen

    def plan(self, goal: str) -> list[dict]:
        return parse_dag(self.generate(goal))


def parse_dag(text: str) -> list[dict]:
    """Extract the DAG node list from model output."""
    if not text:
        return []
    m = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        obj = json.loads(text[start:end + 1])
        return obj.get("dag", []) if isinstance(obj, dict) else []
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    import argparse
    from modelscope import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="清洗这批病理文档、切块并抽取实体和关系，最后做三元组校验")
    ap.add_argument("--base", default=None)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    args = ap.parse_args()
    base = args.base or snapshot_download("Qwen/Qwen3.5-0.8B")
    orch = LocalOrchestrator(base, args.adapter)
    print(json.dumps(orch.plan(args.goal), ensure_ascii=False, indent=2))
