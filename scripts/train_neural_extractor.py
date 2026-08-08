"""Train the neural GPLinker joint entity-relation extractor on CMeIE-V2.

Run inside the ML env (torch + transformers + GPU):

    python scripts/train_neural_extractor.py \
        --data_dir ../CMeIE-V2 \
        --encoder hfl/chinese-roberta-wwm-ext \
        --epochs 20 --batch_size 8 --max_length 256

Produces a checkpoint under data/models/neural_extractor/ with:
  pytorch_model.bin, extractor_config.json, tokenizer files, train_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medigraph.schema.cmeie_schema import CMEIE_ENTITY_TYPES, CMEIE_PREDICATE_KEYS  # noqa: E402
from medigraph.schema.normalize import canonical_key  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from medigraph.extraction.neural_gplinker import _build_torch_modules  # noqa: E402

ENTITY_TYPES = sorted(set(CMEIE_ENTITY_TYPES.values()))
ENTITY2ID = {t: i for i, t in enumerate(ENTITY_TYPES)}


def obj_value(value):
    if isinstance(value, dict):
        return value.get("@value", "")
    return value


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_predicate_space(records: list[dict]) -> list[str]:
    preds = set()
    for rec in records:
        for spo in rec.get("spo_list", []):
            p = str(spo.get("predicate", "")).strip()
            if p:
                preds.add(p)
    return sorted(preds)


def find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    if not needle:
        return out
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return out
        out.append(idx)
        start = idx + 1


class CMeIEDataset(Dataset):
    def __init__(self, records, tokenizer, pred2id, max_length):
        self.records = records
        self.tok = tokenizer
        self.pred2id = pred2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        text = str(rec.get("text", ""))
        enc = self.tok(text, return_offsets_mapping=True, max_length=self.max_length,
                       truncation=True)
        offsets = enc["offset_mapping"]
        start_of = {}
        end_of = {}
        for ti, (cs, ce) in enumerate(offsets):
            if cs == ce:  # special tokens
                continue
            start_of.setdefault(cs, ti)
            end_of[ce] = ti

        def char_to_tok(cs, ce):
            return start_of.get(cs), end_of.get(ce)

        ent_set = set()       # (type_id, ts, te)
        head_set = set()      # (pred_id, sh, oh)
        tail_set = set()      # (pred_id, st, ot)
        for spo in rec.get("spo_list", []):
            subj = str(spo.get("subject", "")).strip()
            obj = str(obj_value(spo.get("object", ""))).strip()
            st_type = CMEIE_ENTITY_TYPES.get(str(spo.get("subject_type", "")), "")
            ot_raw = spo.get("object_type", {})
            ot_type = CMEIE_ENTITY_TYPES.get(str(obj_value(ot_raw)), "")
            predicate = str(spo.get("predicate", "")).strip()
            if not subj or not obj or predicate not in self.pred2id:
                continue
            subj_spans, obj_spans = [], []
            for cs in find_all(text, subj):
                t0, t1 = char_to_tok(cs, cs + len(subj))
                if t0 is not None and t1 is not None and t0 <= t1:
                    subj_spans.append((t0, t1))
                    if st_type in ENTITY2ID:
                        ent_set.add((ENTITY2ID[st_type], t0, t1))
            for cs in find_all(text, obj):
                t0, t1 = char_to_tok(cs, cs + len(obj))
                if t0 is not None and t1 is not None and t0 <= t1:
                    obj_spans.append((t0, t1))
                    if ot_type in ENTITY2ID:
                        ent_set.add((ENTITY2ID[ot_type], t0, t1))
            pid = self.pred2id[predicate]
            for (sh, st) in subj_spans:
                for (oh, ot) in obj_spans:
                    head_set.add((pid, sh, oh))
                    tail_set.add((pid, st, ot))
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
            "length": len(enc["input_ids"]),
            "entities": list(ent_set),
            "heads": list(head_set),
            "tails": list(tail_set),
        }


def collate(batch, num_entity_types, num_predicates, pad_id):
    maxlen = max(b["length"] for b in batch)
    bs = len(batch)
    input_ids = torch.full((bs, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((bs, maxlen), dtype=torch.long)
    tti = torch.zeros((bs, maxlen), dtype=torch.long)
    ent_y = torch.zeros((bs, num_entity_types, maxlen, maxlen))
    head_y = torch.zeros((bs, num_predicates, maxlen, maxlen))
    tail_y = torch.zeros((bs, num_predicates, maxlen, maxlen))
    for i, b in enumerate(batch):
        n = b["length"]
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attn[i, :n] = torch.tensor(b["attention_mask"])
        tti[i, :n] = torch.tensor(b["token_type_ids"][:n])
        for (t, s, e) in b["entities"]:
            if s < maxlen and e < maxlen:
                ent_y[i, t, s, e] = 1
        for (p, s, e) in b["heads"]:
            if s < maxlen and e < maxlen:
                head_y[i, p, s, e] = 1
        for (p, s, e) in b["tails"]:
            if s < maxlen and e < maxlen:
                tail_y[i, p, s, e] = 1
    return input_ids, attn, tti, ent_y, head_y, tail_y


@torch.no_grad()
def quick_triple_f1(model, loader, device, num_predicates, max_batches=60):
    model.eval()
    tp = fp = fn = 0
    for bi, (ids, attn, tti, ent_y, head_y, tail_y) in enumerate(loader):
        if bi >= max_batches:
            break
        ids, attn = ids.to(device), attn.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            ent_l, head_l, tail_l = model(ids, attn)
        ent_l = ent_l.float().cpu(); head_l = head_l.float().cpu(); tail_l = tail_l.float().cpu()
        for i in range(ids.size(0)):
            spans = set()
            ent_pred = (ent_l[i] > 0).any(dim=0)
            for s, e in ent_pred.nonzero(as_tuple=False).tolist():
                spans.add((s, e))
            pred = set()
            for p in range(num_predicates):
                hp = {(int(a), int(b)) for a, b in (head_l[i, p] > 0).nonzero(as_tuple=False).tolist()}
                if not hp:
                    continue
                tpp = {(int(a), int(b)) for a, b in (tail_l[i, p] > 0).nonzero(as_tuple=False).tolist()}
                for (sh, st) in spans:
                    for (oh, ot) in spans:
                        if (sh, oh) in hp and (st, ot) in tpp:
                            pred.add((sh, st, p, oh, ot))
            gold = set()
            g_head = {(int(p), int(s), int(e)) for p, s, e in (head_y[i] > 0).nonzero(as_tuple=False).tolist()}
            g_tail = {(int(p), int(s), int(e)) for p, s, e in (tail_y[i] > 0).nonzero(as_tuple=False).tolist()}
            for (p, sh, oh) in g_head:
                for (p2, st, ot) in g_tail:
                    if p2 == p:
                        gold.add((sh, st, p, oh, ot))
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def gold_triples_for(rec) -> set:
    out = set()
    text = str(rec.get("text", ""))
    for spo in rec.get("spo_list", []):
        subj = str(spo.get("subject", "")).strip()
        obj = str(obj_value(spo.get("object", ""))).strip()
        predicate = str(spo.get("predicate", "")).strip()
        if subj and obj and predicate:
            out.add((canonical_key(subj), predicate, canonical_key(obj)))
    return out, text


@torch.no_grad()
def real_dev_triple_f1(model, dev_records, tokenizer, predicates, n_ent, device,
                       max_length, limit=600, thr=0.0):
    """True end-to-end triple F1 (entity-gated decode -> text -> canonical key),
    the same metric the benchmark uses, for honest checkpoint selection."""
    model.eval()
    tp = fp = fn = 0
    for rec in dev_records[:limit]:
        gold, text = gold_triples_for(rec)
        if not text.strip():
            continue
        enc = tokenizer(text, return_offsets_mapping=True, max_length=max_length,
                        truncation=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            ent_l, head_l, tail_l = model(enc["input_ids"], enc["attention_mask"])
        ent_l = ent_l[0].float().cpu(); head_l = head_l[0].float().cpu(); tail_l = tail_l[0].float().cpu()
        ent_any = (ent_l > thr).any(dim=0)
        starts = {}
        for s, e in ent_any.nonzero(as_tuple=False).tolist():
            starts.setdefault(s, []).append(e)
        pred = set()
        for p in range(len(predicates)):
            hp = {(int(a), int(b)) for a, b in (head_l[p] > thr).nonzero(as_tuple=False).tolist()}
            if not hp:
                continue
            tset = {(int(a), int(b)) for a, b in (tail_l[p] > thr).nonzero(as_tuple=False).tolist()}
            for (sh, oh) in hp:
                for st in starts.get(sh, ()):
                    for ot in starts.get(oh, ()):
                        if (st, ot) in tset and st < len(offsets) and ot < len(offsets):
                            subj = text[offsets[sh][0]:offsets[st][1]]
                            obj = text[offsets[oh][0]:offsets[ot][1]]
                            if subj.strip() and obj.strip():
                                pred.add((canonical_key(subj), predicates[p], canonical_key(obj)))
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    return {"precision": round(prec, 4), "recall": round(rec_, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(PROJECT_ROOT.parent / "CMeIE-V2"))
    ap.add_argument("--file_tpl", default="CMeIE-V2_{split}.jsonl",
                    help="e.g. 'CMeIE_{split}.jsonl' for CMeIE-V1")
    ap.add_argument("--encoder", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "models" / "neural_extractor"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_train", type=int, default=0, help="debug: cap train size")
    args = ap.parse_args()

    random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_recs = load_records(data_dir / args.file_tpl.format(split="train"))
    dev_recs = load_records(data_dir / args.file_tpl.format(split="dev"))
    if args.max_train:
        train_recs = train_recs[: args.max_train]
        dev_recs = dev_recs[:200]
    predicates = build_predicate_space(train_recs)
    pred2id = {p: i for i, p in enumerate(predicates)}
    print(f"train={len(train_recs)} dev={len(dev_recs)} preds={len(predicates)} "
          f"ent_types={len(ENTITY_TYPES)} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    _, GPLinker, multilabel_cce = _build_torch_modules()
    model = GPLinker(args.encoder, len(ENTITY_TYPES), len(predicates)).to(device)

    pad_id = tokenizer.pad_token_id or 0
    train_ds = CMeIEDataset(train_recs, tokenizer, pred2id, args.max_length)
    dev_ds = CMeIEDataset(dev_recs, tokenizer, pred2id, args.max_length)
    coll = lambda b: collate(b, len(ENTITY_TYPES), len(predicates), pad_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=coll, num_workers=0, drop_last=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    # persist config early so eval can run against partial checkpoints
    schema_map = {}
    for cn, key in CMEIE_PREDICATE_KEYS.items():
        schema_map[cn] = {"key": key}
    cfg = {
        "encoder_name": args.encoder, "entity_types": ENTITY_TYPES,
        "predicates": predicates, "predicate_schema": schema_map,
        "max_length": args.max_length, "max_span": 30,
        "version": "neural-gplinker-1.0.0",
    }
    (out_dir / "extractor_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(out_dir)

    log = {"args": vars(args), "epochs": []}
    best_f1 = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (ids, attn, tti, ent_y, head_y, tail_y) in enumerate(train_dl):
            ids, attn = ids.to(device), attn.to(device)
            ent_y = ent_y.to(device); head_y = head_y.to(device); tail_y = tail_y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                ent_l, head_l, tail_l = model(ids, attn)
                b = ids.size(0)
                le = multilabel_cce(ent_l.reshape(b, len(ENTITY_TYPES), -1).float(),
                                    ent_y.reshape(b, len(ENTITY_TYPES), -1))
                lh = multilabel_cce(head_l.reshape(b, len(predicates), -1).float(),
                                    head_y.reshape(b, len(predicates), -1))
                lt = multilabel_cce(tail_l.reshape(b, len(predicates), -1).float(),
                                    tail_y.reshape(b, len(predicates), -1))
                loss = le + lh + lt
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"ep{epoch} step{step}/{len(train_dl)} loss={loss.item():.3f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        metrics = real_dev_triple_f1(model, dev_recs, tokenizer, predicates,
                                     len(ENTITY_TYPES), device, args.max_length, limit=600)
        avg = running / max(1, len(train_dl))
        print(f"[epoch {epoch}] train_loss={avg:.3f} REAL_dev_triple={metrics}", flush=True)
        log["epochs"].append({"epoch": epoch, "train_loss": round(avg, 4),
                              "dev_triple": metrics, "elapsed_s": round(time.time() - t0)})
        (out_dir / "train_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
            log["best_epoch"] = epoch; log["best_dev_triple_f1"] = best_f1
            (out_dir / "train_log.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  saved best checkpoint (REAL dev triple f1={best_f1})", flush=True)
    print(f"done. best REAL dev triple f1={best_f1}")


if __name__ == "__main__":
    main()
