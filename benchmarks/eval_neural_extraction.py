"""Full-detail evaluation for the trained neural GPLinker extractor.

Scores entities and end-to-end triples with the *identical* metric harness as
``eval_fast_extraction.py`` (same canonicalisation, same CMeIE-V2 gold builder),
so neural and lexicon numbers are directly comparable.

    python benchmarks/eval_neural_extraction.py --split dev
    python benchmarks/eval_neural_extraction.py --split test
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402
from medigraph.schema.normalize import canonical_key  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402
from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor  # noqa: E402
from benchmarks.eval_fast_extraction import (  # noqa: E402
    _load_cmeie, _prf, _percentile, _record_counts, _sha256,
)

CMEIE_DIR = PROJECT_ROOT.parent / "CMeIE-V2"


def evaluate(samples, extractor, threshold: float = 0.0, rel_threshold=None) -> dict:
    entity_totals = [0, 0, 0]
    triple_totals = [0, 0, 0]
    strict_totals = [0, 0, 0]
    entity_labels: dict[str, list[int]] = {}
    relation_labels: dict[str, list[int]] = {}
    latencies, details = [], []
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        result = extractor.extract(sample["text"], threshold=threshold,
                                   rel_threshold=rel_threshold)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

        predicted_entities = {
            (canonical_key(e["name"]), e["type"]) for e in result["entities"]
        }
        predicted_triples = {
            (canonical_key(t["head"]), t["predicate"], canonical_key(t["tail"]))
            for t in result["triples"]
        }
        # strict = exact raw-string (subject, predicate, object) match, no
        # canonicalization -- comparable to the official CMeIE SPO-F1 scorer.
        predicted_strict = {
            (str(t["head"]), t["predicate"], str(t["tail"])) for t in result["triples"]
        }
        gold_strict = sample.get("triples_strict", set())
        strict_counts = (
            len(predicted_strict & gold_strict),
            len(predicted_strict - gold_strict),
            len(gold_strict - predicted_strict),
        )
        for offset, value in enumerate(strict_counts):
            strict_totals[offset] += value
        gold_entities = sample["entities"]
        gold_triples = sample["triples"]

        entity_counts = (
            len(predicted_entities & gold_entities),
            len(predicted_entities - gold_entities),
            len(gold_entities - predicted_entities),
        )
        triple_counts = (
            len(predicted_triples & gold_triples),
            len(predicted_triples - gold_triples),
            len(gold_triples - predicted_triples),
        )
        for offset, value in enumerate(entity_counts):
            entity_totals[offset] += value
        for offset, value in enumerate(triple_counts):
            triple_totals[offset] += value
        for label in {item[1] for item in predicted_entities | gold_entities}:
            _record_counts(entity_labels, label,
                           {i for i in predicted_entities if i[1] == label},
                           {i for i in gold_entities if i[1] == label})
        for label in {item[1] for item in predicted_triples | gold_triples}:
            _record_counts(relation_labels, label,
                           {i for i in predicted_triples if i[1] == label},
                           {i for i in gold_triples if i[1] == label})
        if index % 500 == 0:
            print(f"  evaluated {index}/{len(samples)}", flush=True)
        if index <= 400:
            details.append({
                "sample": index,
                "text_sha256": hashlib.sha256(sample["text"].encode("utf-8")).hexdigest(),
                "latency_ms": round(elapsed_ms, 4),
                "entity": _prf(*entity_counts),
                "triple": _prf(*triple_counts),
            })

    return {
        "samples": len(samples),
        "entity_micro": _prf(*entity_totals),
        "end_to_end_triple_micro": _prf(*triple_totals),
        "end_to_end_triple_micro_strict": _prf(*strict_totals),
        "entity_by_type": {k: _prf(*v) for k, v in sorted(entity_labels.items())},
        "relation_by_label": {k: _prf(*v) for k, v in sorted(relation_labels.items())},
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.5), 4),
            "p95": round(_percentile(latencies, 0.95), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--model_dir", default=str(PROJECT_ROOT / "data" / "models" / "neural_extractor"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--rel_threshold", type=float, default=None)
    ap.add_argument("--data_dir", default=str(CMEIE_DIR))
    ap.add_argument("--file_tpl", default="CMeIE-V2_{split}.jsonl",
                    help="e.g. 'CMeIE_{split}.jsonl' for CMeIE-V1")
    ap.add_argument("--tag", default="", help="output filename tag, e.g. 'v1'")
    args = ap.parse_args()

    src = Path(args.data_dir) / args.file_tpl.format(split=args.split)
    samples = _load_cmeie(src)
    # attach raw strict gold (order-preserving parallel read)
    import json as _json
    raw = [_json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    for s, r in zip(samples, raw):
        strict = set()
        for spo in r.get("spo_list", []):
            subj = str(spo.get("subject", "")).strip()
            ov = spo.get("object", {})
            obj = str(ov.get("@value", "") if isinstance(ov, dict) else ov).strip()
            pred = str(spo.get("predicate", ""))
            if subj and obj and pred:
                strict.add((subj, pred, obj))
        s["triples_strict"] = strict
    if args.limit:
        samples = samples[: args.limit]
    extractor = NeuralGPLinkerExtractor(args.model_dir)
    print(f"neural eval split={args.split} samples={len(samples)} device={extractor.device}",
          flush=True)
    report = evaluate(samples, extractor, threshold=args.threshold,
                      rel_threshold=args.rel_threshold)
    report.update({
        "decode_threshold": args.threshold,
        "rel_threshold": args.rel_threshold if args.rel_threshold is not None else args.threshold,
        "mode": "cmeie_neural",
        "split": args.split,
        "extractor": "neural_gplinker",
        "model_version": extractor.version,
        "encoder": extractor.encoder_name,
        # Recorded because `latency_ms` is meaningless without it: runs of this
        # script on cpu and on cuda were once tabulated side by side as if they
        # were a model-size comparison, which made the larger model look four
        # times faster than the base one.
        "device": str(extractor.device),
        "source_file": str(src),
        "source_sha256": _sha256(src),
    })
    tag = f"_{args.tag}" if args.tag else ""
    out = OUTPUTS_DIR / f"eval_neural_cmeie{tag}_{args.split}.json"
    write_json(report, out)
    print(f"entity_micro={report['entity_micro']}")
    print(f"triple_micro (lenient)={report['end_to_end_triple_micro']}")
    print(f"triple_micro (STRICT) ={report['end_to_end_triple_micro_strict']}")
    print(f"latency_ms={report['latency_ms']}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
