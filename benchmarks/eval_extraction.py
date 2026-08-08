"""Extraction-quality evaluation: NER & RE precision / recall / F1 on a gold set.

Runs medical_ner + medical_re over the gold set (CMeIE-V2 + DiaKG, mapped to our
ontology) and scores against the labels under two matching modes:

  * strict  — exact name match after normalization (case/space-insensitive via
              canonical_key), same entity type.
  * relaxed — boundary-tolerant: same type and one span contains the other
              (bidirectional containment). This is the standard clinical-NER
              "relaxed/partial" match (cf. i2b2). It is the fair headline metric
              here because public CMeIE/DiaKG gold is RE-driven and *sparse*
              (only relation-participating entities are labelled, and many spans
              are long descriptive clauses), so exact micro-precision is
              artificially depressed by correct-but-unlabelled or differently-
              segmented entities.

We also report Recall prominently (coverage of gold entities/triples), which is
robust to the gold's sparsity.

Usage:
  python benchmarks/eval_extraction.py [--limit N]
Outputs a table to stdout and writes outputs/eval_extraction.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR  # noqa: E402
from medigraph.llm.client import LLMClient  # noqa: E402
from medigraph.operators.medical_ner import MedicalNEROperator  # noqa: E402
from medigraph.operators.medical_re import MedicalREOperator  # noqa: E402
from medigraph.schema.normalize import canonical_key  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

enable_utf8()
_DATA_GOLD = Path(__file__).resolve().parents[1] / "data" / "gold"
# Prefer the real benchmark gold (CMeIE-V2 + DiaKG, mapped); fall back to the small hand-set.
_REAL = _DATA_GOLD / "ner_re_gold.json"
GOLD = _REAL if _REAL.exists() else (Path(__file__).resolve().parent / "gold_eval_set.json")


def _resolve_gold(arg: str) -> Path:
    """--gold accepts: 'public' (CMeIE+DiaKG), 'cm3kg' (controlled in-domain),
    'pathology' (full-ontology coverage probe), or a path."""
    if not arg or arg == "public":
        return GOLD
    if arg == "cm3kg":
        return _DATA_GOLD / "cm3kg_gold.json"
    if arg == "pathology":
        return _DATA_GOLD / "pathology_probe_gold.json"
    return Path(arg)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _ent_set(items: list[dict]) -> set:
    return {(canonical_key(e["name"]), e["type"]) for e in items if e.get("name")}


def _triple_set(items: list[dict]) -> set:
    # match on (head, relation, tail), normalization-aware; type-agnostic so a
    # correct relation is not penalized for a head/tail type disagreement
    return {
        (canonical_key(t["head"]), t["relation"], canonical_key(t["tail"]))
        for t in items if t.get("head") and t.get("tail") and t.get("relation")
    }


def _contains(a: str, b: str) -> bool:
    """Bidirectional containment on normalized strings (boundary-tolerant)."""
    return a == b or (len(a) >= 2 and len(b) >= 2 and (a in b or b in a))


def _relaxed_counts(pred: set, gold: set, key) -> tuple[int, int, int]:
    """Greedy 1-1 relaxed matching. `key` extracts the comparable signature; two
    items match if their non-string fields are equal and their string field(s)
    satisfy bidirectional containment. Returns (tp, fp, fn)."""
    gold_left = list(gold)
    tp = 0
    used = [False] * len(gold_left)
    for p in pred:
        for j, gld in enumerate(gold_left):
            if used[j]:
                continue
            if key(p, gld):
                used[j] = True
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gold_left) - tp
    return tp, fp, fn


def _ent_relaxed_key(p, g) -> bool:
    return p[1] == g[1] and _contains(p[0], g[0])


def _tri_relaxed_key(p, g) -> bool:
    return p[1] == g[1] and _contains(p[0], g[0]) and _contains(p[2], g[2])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N gold samples (0 = all)")
    ap.add_argument("--gold", default="public",
                    help="'public' (CMeIE+DiaKG), 'cm3kg' (controlled in-domain), or a path")
    ap.add_argument("--out", default="eval_extraction.json", help="output json filename under outputs/")
    args = ap.parse_args()
    gold_path = _resolve_gold(args.gold)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))["samples"]
    if args.limit:
        gold = gold[: args.limit]
    llm = LLMClient()
    ner = MedicalNEROperator(llm=llm)
    re_op = MedicalREOperator(llm=llm)
    print(f"Evaluating on {len(gold)} gold samples with model {llm.config.model} ...\n")

    # strict and relaxed accumulators
    acc = {k: [0, 0, 0] for k in ("ner_s", "ner_r", "re_s", "re_r")}  # [tp, fp, fn]
    by_src: dict[str, dict] = {}  # source -> {ner_r:[...], re_r:[...]} (relaxed, headline)
    per_sample = []

    def _add(key, tpl):
        acc[key][0] += tpl[0]; acc[key][1] += tpl[1]; acc[key][2] += tpl[2]

    def _add_src(src, key, tpl):
        d = by_src.setdefault(src, {"ner_r": [0, 0, 0], "re_r": [0, 0, 0]})
        d[key][0] += tpl[0]; d[key][1] += tpl[1]; d[key][2] += tpl[2]

    for i, s in enumerate(gold, 1):
        text = s["text"]
        gold_ents = _ent_set(s.get("entities", []))
        gold_tri = _triple_set(s.get("triples", []))

        pred_ents_raw = ner.run({"text": text}).get("entities", [])
        pred_ents = _ent_set(pred_ents_raw)
        pred_tri = _triple_set(re_op.run({"text": text, "entities": pred_ents_raw}).get("triples", []))

        # strict (exact-after-normalization set overlap)
        es = (len(pred_ents & gold_ents), len(pred_ents - gold_ents), len(gold_ents - pred_ents))
        ts = (len(pred_tri & gold_tri), len(pred_tri - gold_tri), len(gold_tri - pred_tri))
        # relaxed (boundary-tolerant greedy match)
        er = _relaxed_counts(pred_ents, gold_ents, _ent_relaxed_key)
        tr = _relaxed_counts(pred_tri, gold_tri, _tri_relaxed_key)

        _add("ner_s", es); _add("re_s", ts); _add("ner_r", er); _add("re_r", tr)
        src = s.get("source", "?")
        _add_src(src, "ner_r", er); _add_src(src, "re_r", tr)
        per_sample.append({"sample": i, "source": src,
                           "ner_strict": _prf(*es), "ner_relaxed": _prf(*er),
                           "re_strict": _prf(*ts), "re_relaxed": _prf(*tr)})
        print(f"  sample {i}: NER F1 strict={_prf(*es)['f1']} relaxed={_prf(*er)['f1']}  "
              f"RE F1 strict={_prf(*ts)['f1']} relaxed={_prf(*tr)['f1']}")

    ner_s = _prf(*acc["ner_s"]); ner_r = _prf(*acc["ner_r"])
    re_s = _prf(*acc["re_s"]); re_r = _prf(*acc["re_r"])
    report = {
        "model": llm.config.model,
        "samples": len(gold),
        "gold": gold_path.name,
        "ner_strict": ner_s, "ner_relaxed": ner_r,
        "re_strict": re_s, "re_relaxed": re_r,
        "by_source": {src: {"ner_relaxed": _prf(*d["ner_r"]), "re_relaxed": _prf(*d["re_r"])}
                      for src, d in by_src.items()},
        # back-compat headline keys (relaxed = fair public-benchmark headline)
        "ner": ner_r, "re": re_r,
        "per_sample": per_sample,
        "llm_stats": llm.stats.summary(),
    }
    write_json(report, OUTPUTS_DIR / args.out)

    print(f"\n================ EXTRACTION QUALITY ({gold_path.name}) ================")
    print("                 Precision   Recall      F1")
    print(f"  NER  strict     {ner_s['precision']:<11}{ner_s['recall']:<11}{ner_s['f1']}")
    print(f"  NER  relaxed    {ner_r['precision']:<11}{ner_r['recall']:<11}{ner_r['f1']}")
    print(f"  RE   strict     {re_s['precision']:<11}{re_s['recall']:<11}{re_s['f1']}")
    print(f"  RE   relaxed    {re_r['precision']:<11}{re_r['recall']:<11}{re_r['f1']}")
    if len(by_src) > 1:
        print("\n  -- relaxed F1 by source --")
        for src, d in sorted(by_src.items()):
            print(f"     {src:<12} NER F1={_prf(*d['ner_r'])['f1']}  RE F1={_prf(*d['re_r'])['f1']}")
    print(f"\n  headline (relaxed): NER F1={ner_r['f1']}  RE F1={re_r['f1']}")
    print(f"  (saved -> {OUTPUTS_DIR / args.out})")


if __name__ == "__main__":
    main()
