"""Two-model ensemble evaluation (roberta-base + macbert-large).

Diverse encoders make different errors; merging their predictions is a standard,
honest way to lift F1.  We evaluate three fusion policies on CMeIE-V2 dev with
the same harness as the single-model benchmark and report the best.

    python benchmarks/eval_ensemble.py --limit 800
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402
from medigraph.schema.normalize import canonical_key  # noqa: E402
from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402
from benchmarks.eval_fast_extraction import _load_cmeie, _prf  # noqa: E402

CMEIE_DIR = PROJECT_ROOT.parent / "CMeIE-V2"


def triple_keys(triples):
    return {(canonical_key(t["head"]), t["predicate"], canonical_key(t["tail"])): t
            for t in triples}


def entity_keys(ents):
    return {(canonical_key(e["name"]), e["type"]) for e in ents}


def score(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    return tp, len(pred_set - gold_set), len(gold_set - pred_set)


def raw_strict_keys(triples):
    return {(str(t["head"]), t["predicate"], str(t["tail"])) for t in triples}


def strict_gold(path):
    import json
    out = []
    for line in Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = set()
        for spo in r.get("spo_list", []):
            subj = str(spo.get("subject", "")).strip()
            ov = spo.get("object", {})
            obj = str(ov.get("@value", "") if isinstance(ov, dict) else ov).strip()
            pred = str(spo.get("predicate", ""))
            if subj and obj and pred:
                s.add((subj, pred, obj))
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_conf", type=float, default=0.75,
                    help="confidence floor for single-model-only predictions")
    ap.add_argument("--data_dir", default=str(CMEIE_DIR))
    ap.add_argument("--file_tpl", default="CMeIE-V2_{split}.jsonl")
    ap.add_argument("--m1_dir", default="data/models/neural_extractor_base")
    ap.add_argument("--m2_dir", default="data/models/neural_extractor")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    src = Path(args.data_dir) / args.file_tpl.format(split="dev")
    samples = _load_cmeie(src)
    gold_strict = strict_gold(src)
    if args.limit:
        samples = samples[: args.limit]
        gold_strict = gold_strict[: args.limit]

    m1 = NeuralGPLinkerExtractor(PROJECT_ROOT / args.m1_dir)
    m2 = NeuralGPLinkerExtractor(PROJECT_ROOT / args.m2_dir)
    print(f"ensemble eval samples={len(samples)} src={src.name}", flush=True)

    policies = {"union": [0, 0, 0], "intersection": [0, 0, 0], "conf_union": [0, 0, 0]}
    strict_conf_union = [0, 0, 0]
    ent_tot = [0, 0, 0]
    latencies = []
    for i, s in enumerate(samples, 1):
        import time
        t = time.perf_counter()
        r1 = m1.extract(s["text"]); r2 = m2.extract(s["text"])
        latencies.append((time.perf_counter() - t) * 1000)
        t1, t2 = triple_keys(r1["triples"]), triple_keys(r2["triples"])
        k1, k2 = set(t1), set(t2)
        union = k1 | k2
        inter = k1 & k2
        conf_union = inter | {k for k in (k1 ^ k2)
                              if (t1.get(k) or t2.get(k)).get("confidence", 0) >= args.min_conf}
        for name, keys in (("union", union), ("intersection", inter),
                           ("conf_union", conf_union)):
            tp, fp, fn = score(keys, s["triples"])
            policies[name][0] += tp; policies[name][1] += fp; policies[name][2] += fn
        # strict (raw string) conf_union, comparable to official SPO-F1
        s1, s2 = raw_strict_keys(r1["triples"]), raw_strict_keys(r2["triples"])
        sinter = s1 & s2
        # map raw keys' confidence via the triple dicts
        rc1 = {(str(t["head"]), t["predicate"], str(t["tail"])): t for t in r1["triples"]}
        rc2 = {(str(t["head"]), t["predicate"], str(t["tail"])): t for t in r2["triples"]}
        sconf = sinter | {k for k in (s1 ^ s2)
                          if (rc1.get(k) or rc2.get(k)).get("confidence", 0) >= args.min_conf}
        tp, fp, fn = score(sconf, gold_strict[i - 1])
        strict_conf_union[0] += tp; strict_conf_union[1] += fp; strict_conf_union[2] += fn
        e = entity_keys(r1["entities"]) | entity_keys(r2["entities"])
        tp, fp, fn = score(e, s["entities"])
        ent_tot[0] += tp; ent_tot[1] += fp; ent_tot[2] += fn
        if i % 400 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    report = {
        "samples": len(samples),
        "dataset": src.name,
        "models": [args.m1_dir, args.m2_dir],
        "entity_micro_union": _prf(*ent_tot),
        "triple_micro": {name: _prf(*counts) for name, counts in policies.items()},
        "triple_micro_strict_conf_union": _prf(*strict_conf_union),
        "latency_ms_p50": round(statistics.median(latencies), 3),
        "min_conf": args.min_conf,
    }
    tag = f"_{args.tag}" if args.tag else ""
    write_json(report, OUTPUTS_DIR / f"eval_ensemble_cmeie{tag}_dev.json")
    print("entity(union):", report["entity_micro_union"])
    for name, v in report["triple_micro"].items():
        print(f"triple({name}) lenient:", v)
    print("triple(conf_union) STRICT:", report["triple_micro_strict_conf_union"])


if __name__ == "__main__":
    main()
