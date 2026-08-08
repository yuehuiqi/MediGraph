"""Generate official-format CMeIE test predictions for online (Tianchi/CBLUE)
submission, so the model can be scored on the *official held-out test set*
(the dataset the public GPLinker/CasRel/SOTA numbers use), removing the
dev-vs-test caveat.

    python scripts/predict_cmeie_test.py --data_dir ../CMeIE-V1 \
        --file_tpl "CMeIE_{split}.jsonl" --model_dir data/models/neural_extractor_v1

Output: outputs/CMeIE_test_pred.jsonl  (subject/subject_type/predicate/object/object_type),
ready to zip and submit. Entity types per predicate are taken from the most
frequent (subject_type, object_type) pair observed in train.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor  # noqa: E402


def obj_value(v):
    return v.get("@value", "") if isinstance(v, dict) else v


def predicate_types(train_path: Path) -> dict[str, tuple[str, str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            for spo in json.loads(line).get("spo_list", []):
                p = str(spo.get("predicate", ""))
                st = str(spo.get("subject_type", ""))
                ot = str(obj_value(spo.get("object_type", "")))
                if p:
                    counts[p][(st, ot)] += 1
    return {p: c.most_common(1)[0][0] for p, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default=str(Path(__file__).resolve().parents[2] / "CMeIE-V1"))
    ap.add_argument("--file_tpl", default="CMeIE_{split}.jsonl")
    ap.add_argument("--model_dir", default="data/models/neural_extractor_v1")
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "CMeIE_test_pred.jsonl"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ptypes = predicate_types(data_dir / args.file_tpl.format(split="train"))
    extractor = NeuralGPLinkerExtractor(args.model_dir)
    test_path = data_dir / args.file_tpl.format(split="test")
    rows = [json.loads(l) for l in test_path.open(encoding="utf-8") if l.strip()]
    print(f"predicting {len(rows)} test rows on {extractor.device}", flush=True)

    out_path = Path(args.out)
    n_triples = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(rows, 1):
            text = str(rec.get("text", ""))
            result = extractor.extract(text)
            spo_list = []
            seen = set()
            for t in result["triples"]:
                p = t["predicate"]
                st, ot = ptypes.get(p, ("", ""))
                key = (t["head"], p, t["tail"])
                if key in seen:
                    continue
                seen.add(key)
                spo_list.append({
                    "Combined": False,
                    "predicate": p,
                    "subject": t["head"],
                    "subject_type": st,
                    "object": {"@value": t["tail"]},
                    "object_type": {"@value": ot},
                })
            n_triples += len(spo_list)
            out.write(json.dumps({"text": text, "spo_list": spo_list}, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    print(f"wrote {out_path} ({n_triples} triples, avg {n_triples/max(1,len(rows)):.2f}/doc)")
    print("提交：将该文件按官方要求命名/打包后提交 Tianchi CMeIE 榜单以获取官方 test SPO-F1。")


if __name__ == "__main__":
    main()
