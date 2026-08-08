"""Entity-linking accuracy evaluation for the CM3KG-backed EntityLinker.

We measure three real capabilities with a held-out, non-trivial protocol:

  1. Normalization-variant linking: surface perturbations that should collapse
     to the same canonical id (whitespace / full-width <-> half-width / casing /
     trailing punctuation).  Tests the canonical-key normalization.
  2. Typo-variant linking (fuzzy): single-character edits that must be resolved
     by the conservative fuzzy matcher to the correct canonical id.
  3. NIL rejection: non-KB strings (document-structure noise + random tokens)
     must NOT be falsely linked (match_method == unlinked_local_id).

Overall accuracy = (correct in-KB links + correct NIL rejections) / total.

    python benchmarks/eval_entity_linking.py
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ENTITY_LINKER_ARTIFACT, OUTPUTS_DIR  # noqa: E402
from medigraph.extraction.entity_linker import EntityLinker  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

random.seed(42)

_FULL = "０１２３４５６７８９ＡＢＣ（）；，"
_HALF = "0123456789ABC();,"
_F2H = str.maketrans(_FULL, _HALF)
_H2F = str.maketrans(_HALF, _FULL)

NOISE = [
    "definition", "clinical features", "radiology images", "gross description",
    "treatment", "see also", "references", "table 1", "figure 2", "section",
    "上一页", "下一页", "参考文献", "返回顶部", "目录", "图1", "表2", "注意事项",
]


def norm_variants(surface: str) -> list[str]:
    out = []
    out.append(f"  {surface} ")                      # whitespace
    out.append(surface.translate(_H2F))              # half -> full width
    if surface.translate(_F2H) != surface:
        out.append(surface.translate(_F2H))          # full -> half width
    out.append(surface + "。")                        # trailing punctuation
    if any(c.isalpha() and c.isascii() for c in surface):
        out.append(surface.upper())
    return [v for v in out if v.strip() and v != surface]


def typo_variants(surface: str) -> list[str]:
    s = surface
    out = []
    if len(s) >= 4:
        i = random.randrange(1, len(s) - 1)
        out.append(s[:i] + s[i + 1:])                # deletion
        j = random.randrange(0, len(s))
        out.append(s[:j] + s[j] + s[j:])             # duplication
    return out


def main():
    linker = EntityLinker.load(ENTITY_LINKER_ARTIFACT)
    entries = list(linker.entries.values())
    random.shuffle(entries)
    sample = entries[:2000]

    norm_tp = norm_n = 0
    typo_tp = typo_n = 0
    errors = []
    for entry in sample:
        gold_id = entry.canonical_id
        for variant in norm_variants(entry.name):
            norm_n += 1
            res = linker.link(variant, entry.type)
            if res["canonical_id"] == gold_id:
                norm_tp += 1
            elif len(errors) < 80:
                errors.append({"kind": "norm", "surface": variant,
                               "gold": entry.name, "got": res["canonical_name"],
                               "method": res["match_method"]})
        for variant in typo_variants(entry.name):
            typo_n += 1
            res = linker.link(variant, entry.type)
            if res["canonical_id"] == gold_id:
                typo_tp += 1

    nil_ok = nil_n = 0
    negatives = list(NOISE)
    for _ in range(400):
        negatives.append(hashlib.sha1(str(random.random()).encode()).hexdigest()[:8])
    for neg in negatives:
        nil_n += 1
        res = linker.link(neg, "")
        if res["match_method"] == "unlinked_local_id":
            nil_ok += 1

    in_kb_tp = norm_tp + typo_tp
    in_kb_n = norm_n + typo_n
    overall = (in_kb_tp + nil_ok) / (in_kb_n + nil_n) if (in_kb_n + nil_n) else 0.0
    report = {
        "kb_entries": len(entries),
        "sampled_entities": len(sample),
        "normalization_variant_accuracy": round(norm_tp / norm_n, 4) if norm_n else 0.0,
        "typo_variant_accuracy_fuzzy": round(typo_tp / typo_n, 4) if typo_n else 0.0,
        "in_kb_linking_accuracy": round(in_kb_tp / in_kb_n, 4) if in_kb_n else 0.0,
        "nil_rejection_rate": round(nil_ok / nil_n, 4) if nil_n else 0.0,
        "overall_accuracy": round(overall, 4),
        "counts": {"norm": [norm_tp, norm_n], "typo": [typo_tp, typo_n],
                   "nil": [nil_ok, nil_n]},
        "error_examples": errors[:40],
    }
    write_json(report, OUTPUTS_DIR / "eval_entity_linking.json")
    for k in ("normalization_variant_accuracy", "typo_variant_accuracy_fuzzy",
              "in_kb_linking_accuracy", "nil_rejection_rate", "overall_accuracy"):
        print(f"{k}: {report[k]}")
    print(f"written {OUTPUTS_DIR / 'eval_entity_linking.json'}")


if __name__ == "__main__":
    main()
