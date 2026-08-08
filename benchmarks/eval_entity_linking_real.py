"""Non-circular entity-linking evaluation on REAL mentions.

The perturbation test (eval_entity_linking.py) measures robustness against the
KB's own surfaces. This complementary test takes mentions the neural NER predicts
on held-out CMeIE-V2 dev text (unseen at KB-build time) and links them to the
linking KB, reporting how many real mentions link and by which method -- so the
linker is exercised on genuine model output, not on self-generated variants.

The KB is `data/models/entity_linker.json` (CM3KG + CMeIE-V2 train + DIAKG). An
earlier revision linked against CM3KG alone, and the label written into the
report still said so; the reported 0.733 was measured against the expanded KB.

    python benchmarks/eval_entity_linking_real.py --limit 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ENTITY_LINKER_ARTIFACT, OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402
from medigraph.extraction.entity_linker import EntityLinker  # noqa: E402
from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

CMEIE_DEV = PROJECT_ROOT.parent / "CMeIE-V2" / "CMeIE-V2_dev.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--model_dir", default=str(PROJECT_ROOT / "data" / "models" / "neural_extractor"))
    args = ap.parse_args()

    linker = EntityLinker.load(ENTITY_LINKER_ARTIFACT)
    extractor = NeuralGPLinkerExtractor(args.model_dir)
    rows = [json.loads(l) for l in CMEIE_DEV.open(encoding="utf-8") if l.strip()][: args.limit]
    print(f"linking real mentions from {len(rows)} dev docs", flush=True)

    total = 0
    methods: Counter = Counter()
    linked_examples = []
    seen = set()
    for rec in rows:
        result = extractor.extract(str(rec.get("text", "")))
        for e in result["entities"]:
            name = str(e.get("name", "")).strip()
            key = (name, e.get("type", ""))
            if not name or key in seen:
                continue
            seen.add(key)
            total += 1
            res = linker.link(name, str(e.get("type", "")))
            methods[res["match_method"]] += 1
            if res["match_method"] != "unlinked_local_id" and len(linked_examples) < 40:
                linked_examples.append({"mention": name, "canonical": res["canonical_name"],
                                        "method": res["match_method"], "score": res["link_score"]})

    linked = total - methods.get("unlinked_local_id", 0)
    report = {
        "protocol": ("real mentions from neural NER on held-out CMeIE-V2 dev, "
                     "linked to data/models/entity_linker.json (CM3KG + CMeIE-V2 train + DIAKG)"),
        "distinct_mentions": total,
        "linked": linked,
        "linked_rate": round(linked / total, 4) if total else 0.0,
        "method_breakdown": {m: methods[m] for m in sorted(methods)},
        "note": "coverage metric (no gold links); precision is covered by eval_entity_linking.py",
        "examples": linked_examples,
    }
    write_json(report, OUTPUTS_DIR / "eval_entity_linking_real.json")
    print(f"distinct real mentions={total}  linked={linked}  linked_rate={report['linked_rate']}")
    print(f"method_breakdown={report['method_breakdown']}")


if __name__ == "__main__":
    main()
