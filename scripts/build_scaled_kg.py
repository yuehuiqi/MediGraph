"""Build a large, self-produced medical knowledge graph with the trained neural
extractor (no third-party graph import).

Every node/edge is produced by GPLinker extraction over raw corpora, linked to
canonical CM3KG ids, and carries provenance (source doc + extractor + score).
A final incremental-revision pass demonstrates auditable graph evolution.

    python scripts/build_scaled_kg.py            # full build
    python scripts/build_scaled_kg.py --limit 500  # quick smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ENTITY_LINKER_ARTIFACT, OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402
from medigraph.extraction.entity_linker import EntityLinker  # noqa: E402
from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor  # noqa: E402
from medigraph.graph.local_store import LocalGraphStore  # noqa: E402
from medigraph.schema.cmeie_schema import CMEIE_PREDICATE_KEYS  # noqa: E402
from medigraph.utils.io import write_json  # noqa: E402

CMEIE_DIR = PROJECT_ROOT.parent / "CMeIE-V2"
DIAKG_DIR = PROJECT_ROOT.parent / "DIAKG" / "0521_new_format"


def cmeie_texts(limit: int = 0) -> list[tuple[str, str]]:
    out = []
    for split in ("train", "dev", "test"):
        path = CMEIE_DIR / f"CMeIE-V2_{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = str(rec.get("text", "")).strip()
                if text:
                    out.append((text, f"CMeIE-V2_{split}"))
    if limit:
        out = out[:limit]
    return out


def diakg_texts() -> list[tuple[str, str]]:
    out = []
    if not DIAKG_DIR.exists():
        return out
    for path in sorted(DIAKG_DIR.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for para in doc.get("paragraphs", []):
            for sent in para.get("sentences", []):
                text = str(sent.get("sentence", "")).strip()
                if len(text) >= 8:
                    out.append((text, f"DIAKG/{path.name}"))
    return out


def ingest(store: LocalGraphStore, extractor, docs, source_tag_default=""):
    triples = 0
    for text, source in docs:
        result = extractor.extract(text)
        type_of = {e["name"]: e["type"] for e in result["entities"]}
        for e in result["entities"]:
            store.upsert_entity(e["name"], e["type"], confidence=e.get("confidence", 0.0),
                                source=source)
        for t in result["triples"]:
            store.upsert_triple(
                head=t["head"], head_type=type_of.get(t["head"], t.get("head_type") or "Other"),
                relation=t.get("relation") or f"cmeie:{CMEIE_PREDICATE_KEYS.get(t['predicate'], '')}",
                tail=t["tail"], tail_type=type_of.get(t["tail"], t.get("tail_type") or "Other"),
                confidence=t.get("confidence", 0.0), source=source,
                operator_version=t.get("model_version", "neural_gplinker"),
                extractor="neural_gplinker", relation_zh=t.get("predicate", ""),
                source_doc=source)
            triples += 1
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(PROJECT_ROOT / "data" / "models" / "neural_extractor"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "graph_scaled.json"))
    args = ap.parse_args()

    t0 = time.time()
    linker = EntityLinker.load(ENTITY_LINKER_ARTIFACT)
    store = LocalGraphStore(linker=linker)
    extractor = NeuralGPLinkerExtractor(args.model_dir)
    docs = cmeie_texts(args.limit) + (diakg_texts() if not args.limit else [])
    print(f"ingesting {len(docs)} documents on {extractor.device} ...", flush=True)

    raw_triples = ingest(store, extractor, docs)
    base_stats = store.stats()
    print(f"base graph: {base_stats}", flush=True)

    # Incremental evolution demo: a fresh revision over the last DIAKG slice.
    store.begin_revision("incremental-diakg-batch")
    delta_docs = diakg_texts()[:200] if not args.limit else docs[: min(20, len(docs))]
    ingest(store, extractor, delta_docs)
    delta = store.commit_revision()

    store.export_json(args.out)
    stats = store.stats()
    report = {
        "documents_ingested": len(docs),
        "raw_triples_emitted": raw_triples,
        "graph": stats,
        "self_produced": True,
        "third_party_graph_import": False,
        "incremental_revision_delta": delta,
        "provenance_per_edge": True,
        "extractor": extractor.version,
        "elapsed_s": round(time.time() - t0, 1),
        "graph_file": str(args.out),
    }
    write_json(report, OUTPUTS_DIR / "kg_scale_report.json")
    print(json.dumps({"nodes": stats.get("num_entities"),
                      "edges": stats.get("num_triples"),
                      "elapsed_s": report["elapsed_s"]}, ensure_ascii=False))
    print(f"written {args.out} and kg_scale_report.json")


if __name__ == "__main__":
    main()
