"""LoRA fine-tuning of Qwen3.5-0.8B as the data-processing orchestrator.

Self-contained TRL + PEFT script. bf16 (no quantization) fits the 12GB laptop GPU
with large margin for a 0.8B model, and avoids bitsandbytes on Windows/Blackwell.

Run in the training conda env (Python 3.11 + torch cu128):
  python finetune/train_lora.py --epochs 3 --batch 8

Downloads the base model from ModelScope on first run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs" / "qwen3p5-0p8b-orchestrator"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def download_model(model_id: str) -> str:
    """Fetch the base model from ModelScope; return local path."""
    from modelscope import snapshot_download

    path = snapshot_download(model_id)
    print(f"[model] {model_id} -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--model-path", default=None, help="local path; skip download if set")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    model_path = args.model_path or download_model(args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("json", data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "eval": str(DATA_DIR / "eval.jsonl"),
    })

    import inspect
    cfg_kwargs = dict(
        output_dir=str(OUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )
    # Version-tolerant optional args: keep only those SFTConfig actually accepts.
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" in sft_params:
        cfg_kwargs["max_seq_length"] = args.max_seq
    elif "max_length" in sft_params:
        cfg_kwargs["max_length"] = args.max_seq
    if "assistant_only_loss" in sft_params:
        cfg_kwargs["assistant_only_loss"] = True
    cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        peft_config=lora,
        processing_class=tokenizer,
    )
    print(f"Training on {len(ds['train'])} examples, eval {len(ds['eval'])} ...")
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\nLoRA adapter saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
