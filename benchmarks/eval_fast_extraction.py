"""Offline, full-detail evaluation for the deterministic L1 fast extractor.

Examples:
  python benchmarks/eval_fast_extraction.py --mode core
  python benchmarks/eval_fast_extraction.py --mode cmeie --split dev
  python benchmarks/eval_fast_extraction.py --mode cmeie --split test

The CMeIE mode preserves all 53 official schema rows / 44 predicate labels.
Reports include per-sample errors, per-label metrics, checksums and latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    CALIBRATION_ARTIFACT,
    FAST_EXTRACTOR_ARTIFACT,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)
from medigraph.extraction.fast_path import FastSpanRelationExtractor  # noqa: E402
from medigraph.schema.cmeie_schema import CMEIE_ENTITY_TYPES  # noqa: E402
from medigraph.schema.normalize import canonical_key  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _record_counts(bucket: dict[str, list[int]], label: str, predicted: set, gold: set) -> None:
    values = bucket.setdefault(label, [0, 0, 0])
    values[0] += len(predicted & gold)
    values[1] += len(predicted - gold)
    values[2] += len(gold - predicted)


def _load_core(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


def _load_cmeie(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            entities = set()
            triples = set()
            for spo in record.get("spo_list", []):
                subject = str(spo.get("subject", "")).strip()
                subject_type = CMEIE_ENTITY_TYPES.get(str(spo.get("subject_type", "")), "")
                obj_raw = spo.get("object", {})
                obj = str(obj_raw.get("@value", "") if isinstance(obj_raw, dict) else obj_raw).strip()
                obj_type_raw = spo.get("object_type", {})
                obj_type_raw = obj_type_raw.get("@value", "") if isinstance(obj_type_raw, dict) else obj_type_raw
                object_type = CMEIE_ENTITY_TYPES.get(str(obj_type_raw), "")
                predicate = str(spo.get("predicate", ""))
                if subject and subject_type:
                    entities.add((canonical_key(subject), subject_type))
                if obj and object_type:
                    entities.add((canonical_key(obj), object_type))
                if subject and obj and predicate:
                    triples.add((canonical_key(subject), predicate, canonical_key(obj)))
            samples.append(
                {
                    "text": str(record.get("text", "")),
                    "entities": entities,
                    "triples": triples,
                    "source": path.name,
                }
            )
    return samples


def evaluate(samples: list[dict], extractor: FastSpanRelationExtractor, mode: str) -> dict:
    entity_totals = [0, 0, 0]
    triple_totals = [0, 0, 0]
    entity_labels: dict[str, list[int]] = {}
    relation_labels: dict[str, list[int]] = {}
    latencies = []
    details = []
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        if mode == "cmeie":
            predicted_entities_raw = extractor.extract_entities(
                sample["text"],
                include_type_alternatives=True,
            )
            predicted_triples_raw = extractor.extract_cmeie_relations(
                sample["text"],
                predicted_entities_raw,
            )
            predicted_entities = {
                (canonical_key(item["name"]), item["type"])
                for item in predicted_entities_raw
            }
            predicted_triples = {
                (canonical_key(item["head"]), item["predicate"], canonical_key(item["tail"]))
                for item in predicted_triples_raw
            }
            gold_entities = sample["entities"]
            gold_triples = sample["triples"]
        else:
            predicted_entities_raw = extractor.extract_entities(
                sample["text"],
                overlap_policy="maximal",
            )
            predicted_triples_raw = extractor.extract_relations(sample["text"], predicted_entities_raw)
            predicted_entities = {
                (canonical_key(item["name"]), item["type"])
                for item in predicted_entities_raw
            }
            predicted_triples = {
                (canonical_key(item["head"]), item["relation"], canonical_key(item["tail"]))
                for item in predicted_triples_raw
            }
            gold_entities = {
                (canonical_key(item["name"]), item["type"])
                for item in sample.get("entities", [])
            }
            gold_triples = {
                (canonical_key(item["head"]), item["relation"], canonical_key(item["tail"]))
                for item in sample.get("triples", [])
            }
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

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

        all_entity_labels = {item[1] for item in predicted_entities | gold_entities}
        for label in all_entity_labels:
            _record_counts(
                entity_labels,
                label,
                {item for item in predicted_entities if item[1] == label},
                {item for item in gold_entities if item[1] == label},
            )
        all_relation_labels = {item[1] for item in predicted_triples | gold_triples}
        for label in all_relation_labels:
            _record_counts(
                relation_labels,
                label,
                {item for item in predicted_triples if item[1] == label},
                {item for item in gold_triples if item[1] == label},
            )
        details.append(
            {
                "sample": index,
                "source": sample.get("source", ""),
                "text_sha256": hashlib.sha256(sample["text"].encode("utf-8")).hexdigest(),
                "latency_ms": round(elapsed_ms, 4),
                "entity": _prf(*entity_counts),
                "triple": _prf(*triple_counts),
                "entity_fp": sorted([list(item) for item in predicted_entities - gold_entities])[:100],
                "entity_fn": sorted([list(item) for item in gold_entities - predicted_entities])[:100],
                "triple_fp": sorted([list(item) for item in predicted_triples - gold_triples])[:100],
                "triple_fn": sorted([list(item) for item in gold_triples - predicted_triples])[:100],
            }
        )
        if index % 500 == 0:
            print(f"  evaluated {index}/{len(samples)}")

    return {
        "samples": len(samples),
        "entity_micro": _prf(*entity_totals),
        "end_to_end_triple_micro": _prf(*triple_totals),
        "entity_by_type": {
            label: _prf(*values)
            for label, values in sorted(entity_labels.items())
        },
        "relation_by_label": {
            label: _prf(*values)
            for label, values in sorted(relation_labels.items())
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.5), 4),
            "p95": round(_percentile(latencies, 0.95), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("core", "cmeie"), default="core")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.mode == "cmeie":
        source = PROJECT_ROOT.parent / "CMeIE-V2" / f"CMeIE-V2_{args.split}.jsonl"
        samples = _load_cmeie(source)
        default_out = OUTPUTS_DIR / f"eval_fast_cmeie_{args.split}.json"
    else:
        source = PROJECT_ROOT / "data" / "gold" / "ner_re_gold.json"
        samples = _load_core(source)
        default_out = OUTPUTS_DIR / "eval_fast_core.json"
    if args.limit:
        samples = samples[: args.limit]
    extractor = FastSpanRelationExtractor.load(
        FAST_EXTRACTOR_ARTIFACT,
        calibration_path=CALIBRATION_ARTIFACT if CALIBRATION_ARTIFACT.exists() else None,
    )
    print(f"Evaluating {len(samples)} samples in {args.mode} mode ...")
    report = evaluate(samples, extractor, args.mode)
    report.update(
        {
            "mode": args.mode,
            "split": args.split if args.mode == "cmeie" else "project_gold",
            "extractor": "deterministic_non_neural_L1_baseline",
            "artifact_version": extractor.version,
            "artifact_sha256": _sha256(FAST_EXTRACTOR_ARTIFACT),
            "source_file": str(source),
            "source_sha256": _sha256(source),
            "calibration_file": str(CALIBRATION_ARTIFACT) if CALIBRATION_ARTIFACT.exists() else "",
        }
    )
    output = Path(args.out) if args.out else default_out
    write_json(report, output)
    print("Entity:", report["entity_micro"])
    print("E2E triple:", report["end_to_end_triple_micro"])
    print("Latency ms:", report["latency_ms"])
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
