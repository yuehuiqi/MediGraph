"""Fit temperature scaling on CMeIE dev predictions and emit ECE evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    CALIBRATION_ARTIFACT,
    FAST_EXTRACTOR_ARTIFACT,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)
from medigraph.extraction.calibration import (  # noqa: E402
    TemperatureCalibrator,
    expected_calibration_error,
    reliability_bins,
)
from medigraph.extraction.fast_path import FastSpanRelationExtractor  # noqa: E402
from medigraph.schema.cmeie_schema import CMEIE_ENTITY_TYPES  # noqa: E402
from medigraph.schema.normalize import canonical_key  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1200, help="0 = all dev samples")
    args = parser.parse_args()
    dev = PROJECT_ROOT.parent / "CMeIE-V2" / "CMeIE-V2_dev.jsonl"
    extractor = FastSpanRelationExtractor.load(FAST_EXTRACTOR_ARTIFACT)
    confidences: list[float] = []
    labels: list[int] = []
    processed = 0
    with dev.open(encoding="utf-8") as handle:
        for line in handle:
            if args.samples and processed >= args.samples:
                break
            record = json.loads(line)
            gold = set()
            for spo in record.get("spo_list", []):
                subject_type = CMEIE_ENTITY_TYPES.get(str(spo.get("subject_type", "")))
                if subject_type:
                    gold.add((canonical_key(str(spo.get("subject", ""))), subject_type))
                obj = spo.get("object", {})
                obj = obj.get("@value", "") if isinstance(obj, dict) else obj
                object_type = spo.get("object_type", {})
                object_type = object_type.get("@value", "") if isinstance(object_type, dict) else object_type
                object_type = CMEIE_ENTITY_TYPES.get(str(object_type))
                if object_type:
                    gold.add((canonical_key(str(obj)), object_type))
            # Calibrate the production output contract (one best type per
            # boundary), not the diagnostic all-type candidate pool.
            predictions = extractor.extract_entities(
                str(record.get("text", "")),
                overlap_policy="maximal",
            )
            for prediction in predictions:
                key = (canonical_key(prediction["name"]), prediction["type"])
                confidences.append(float(prediction["confidence"]))
                labels.append(int(key in gold))
            processed += 1

    before = expected_calibration_error(confidences, labels)
    calibrator = TemperatureCalibrator().fit(confidences, labels)
    transformed = calibrator.transform(confidences)
    after = expected_calibration_error(transformed, labels)
    calibrator.save(CALIBRATION_ARTIFACT)
    report = {
        "method": "temperature_scaling",
        "fit_split": "CMeIE-V2_dev",
        "samples": processed,
        "predictions": len(confidences),
        "positive_predictions": sum(labels),
        "ece_before": before,
        "ece_after": after,
        "calibrator": calibrator.to_dict(),
        "reliability_before": reliability_bins(confidences, labels),
        "reliability_after": reliability_bins(transformed, labels),
        "artifact_sha256": hashlib.sha256(FAST_EXTRACTOR_ARTIFACT.read_bytes()).hexdigest(),
        "note": "ECE is measured over emitted entity predictions; missed gold spans have no confidence.",
    }
    output = OUTPUTS_DIR / "calibration_report.json"
    write_json(report, output)
    print(f"ECE: {before:.4f} -> {after:.4f}; T={calibrator.temperature:.4f}")
    print(f"Saved model -> {CALIBRATION_ARTIFACT}")
    print(f"Saved report -> {output}")


if __name__ == "__main__":
    main()
