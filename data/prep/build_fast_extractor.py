r"""Build the deterministic L1 extraction and entity-linking artifacts.

Sources (all local):
  - CMeIE-V2 train: supervised entities and faithfully mapped core relations
  - DIAKG: supervised nested spans and faithfully mapped relations
  - CM3KG graph: canonical entity vocabulary and structured facts

The output is sorted and contains source checksums, so repeated builds from the
same source files are byte-for-byte reproducible.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import (  # noqa: E402
    ENTITY_LINKER_ARTIFACT,
    FAST_EXTRACTOR_ARTIFACT,
    KG_GRAPH,
    PROJECT_ROOT,
)
from data.prep.ontology_map import (  # noqa: E402
    CMEIE_ENT,
    CMEIE_REL,
    CMEIE_REL_TAIL_TYPE,
    DIAKG_ENT,
    DIAKG_REL,
)
from medigraph.extraction.entity_linker import EntityLinker, LinkEntry, stable_entity_id  # noqa: E402
from medigraph.schema.cmeie_schema import CMEIE_ENTITY_TYPES, predicate_key  # noqa: E402
from medigraph.schema.normalize import canonical_key, canonical_name, is_valid_entity_name  # noqa: E402

WORKSPACE = PROJECT_ROOT.parent
CMEIE_TRAIN = WORKSPACE / "CMeIE-V2" / "CMeIE-V2_train.jsonl"
DIAKG_DIR = WORKSPACE / "DIAKG" / "0521_new_format"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactBuilder:
    def __init__(self):
        self.entity_counts: dict[str, Counter] = defaultdict(Counter)
        self.entity_sources: dict[tuple[str, str], Counter] = defaultdict(Counter)
        self.fact_counts: Counter = Counter()
        self.fact_sources: dict[tuple[str, str, str, str, str], Counter] = defaultdict(Counter)
        self.benchmark_fact_counts: Counter = Counter()

    def add_entity(self, name: str, entity_type: str, source: str) -> None:
        name = canonical_name(name)
        if not entity_type or not is_valid_entity_name(name):
            return
        key = canonical_key(name)
        if len(key) < 2 and entity_type not in {"Gene", "Biomarker"}:
            return
        self.entity_counts[key][(name, entity_type)] += 1
        self.entity_sources[(key, entity_type)][source] += 1

    def add_fact(
        self,
        head: str,
        head_type: str,
        relation: str,
        tail: str,
        tail_type: str,
        source: str,
    ) -> None:
        self.add_entity(head, head_type, source)
        self.add_entity(tail, tail_type, source)
        key = (
            canonical_key(head),
            relation,
            canonical_key(tail),
            head_type,
            tail_type,
        )
        if (
            key[0]
            and key[1]
            and key[2]
            and key[0] in self.entity_counts
            and key[2] in self.entity_counts
        ):
            self.fact_counts[key] += 1
            self.fact_sources[key][source] += 1

    def add_benchmark_fact(
        self,
        head: str,
        head_type: str,
        predicate: str,
        tail: str,
        tail_type: str,
    ) -> None:
        relation = predicate_key(predicate)
        self.add_entity(head, head_type, "CMeIE-V2_train")
        self.add_entity(tail, tail_type, "CMeIE-V2_train")
        head_key, tail_key = canonical_key(head), canonical_key(tail)
        if (
            relation
            and head_key in self.entity_counts
            and tail_key in self.entity_counts
        ):
            self.benchmark_fact_counts[
                (head_key, relation, tail_key, head_type, tail_type, predicate)
            ] += 1

    def read_cmeie(self, path: Path) -> int:
        records = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records += 1
                for spo in record.get("spo_list", []):
                    subject = str(spo.get("subject", "")).strip()
                    raw_subject_type = str(spo.get("subject_type", ""))
                    subject_type = CMEIE_ENTITY_TYPES.get(raw_subject_type)
                    obj_raw = spo.get("object", {})
                    obj = str(obj_raw.get("@value", "") if isinstance(obj_raw, dict) else obj_raw).strip()
                    obj_type_raw = spo.get("object_type", {})
                    obj_type_raw = obj_type_raw.get("@value", "") if isinstance(obj_type_raw, dict) else obj_type_raw
                    relation = CMEIE_REL.get(str(spo.get("predicate", "")))
                    object_type = CMEIE_ENTITY_TYPES.get(str(obj_type_raw))
                    if subject_type:
                        self.add_entity(subject, subject_type, "CMeIE-V2_train")
                    if object_type:
                        self.add_entity(obj, object_type, "CMeIE-V2_train")
                    if relation and subject_type:
                        core_subject_type = CMEIE_ENT.get(raw_subject_type)
                        core_object_type = CMEIE_ENT.get(str(obj_type_raw)) or CMEIE_REL_TAIL_TYPE.get(relation, "")
                        if core_subject_type and core_object_type:
                            self.add_fact(
                                subject,
                                core_subject_type,
                                relation,
                                obj,
                                core_object_type,
                                "CMeIE-V2_train",
                            )
                    if subject_type and object_type:
                        self.add_benchmark_fact(
                            subject,
                            subject_type,
                            str(spo.get("predicate", "")),
                            obj,
                            object_type,
                        )
        return records

    def read_diakg(self, directory: Path) -> int:
        documents = 0
        for path in sorted(directory.glob("*.json"), key=lambda value: int(value.stem)):
            documents += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            for paragraph in data.get("paragraphs", []):
                for sentence in paragraph.get("sentences", []):
                    id_to_entity = {
                        str(entity.get("entity_id")): entity
                        for entity in sentence.get("entities", [])
                    }
                    for entity in id_to_entity.values():
                        entity_type = DIAKG_ENT.get(str(entity.get("entity_type", "")))
                        if entity_type:
                            self.add_entity(str(entity.get("entity", "")), entity_type, "DIAKG")
                    for relation_record in sentence.get("relations", []):
                        relation = DIAKG_REL.get(str(relation_record.get("relation_type", "")))
                        raw_head = id_to_entity.get(str(relation_record.get("head_entity_id", "")))
                        raw_tail = id_to_entity.get(str(relation_record.get("tail_entity_id", "")))
                        if not relation or not raw_head or not raw_tail:
                            continue
                        # DIAKG relation labels are object_Disease.  The project
                        # ontology uses the disease as the relation head.
                        head_type = DIAKG_ENT.get(str(raw_tail.get("entity_type", "")))
                        tail_type = DIAKG_ENT.get(str(raw_head.get("entity_type", "")))
                        if head_type and tail_type:
                            self.add_fact(
                                str(raw_tail.get("entity", "")),
                                head_type,
                                relation,
                                str(raw_head.get("entity", "")),
                                tail_type,
                                "DIAKG",
                            )
        return documents

    def read_graph(self, path: Path) -> tuple[int, int, list[LinkEntry]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        node_types = {
            str(node.get("id", "")): str(node.get("type", ""))
            for node in data.get("nodes", [])
        }
        link_entries = []
        for name, entity_type in node_types.items():
            self.add_entity(name, entity_type, "CM3KG")
            link_entries.append(
                LinkEntry(
                    canonical_id=stable_entity_id(name, entity_type),
                    name=name,
                    type=entity_type,
                    source="CM3KG",
                )
            )
        for edge in data.get("edges", []):
            head, tail = str(edge.get("head", "")), str(edge.get("tail", ""))
            self.add_fact(
                head,
                str(edge.get("head_type") or node_types.get(head, "")),
                str(edge.get("relation", "")),
                tail,
                str(edge.get("tail_type") or node_types.get(tail, "")),
                "CM3KG",
            )
        return len(node_types), len(data.get("edges", [])), link_entries

    def artifact(self, source_checksums: dict[str, str]) -> dict:
        entities = []
        for key, variants in self.entity_counts.items():
            by_type: dict[str, Counter] = defaultdict(Counter)
            for (name, entity_type), count in variants.items():
                by_type[entity_type][name] += count
            for entity_type, names in by_type.items():
                name, count = names.most_common(1)[0]
                source = self.entity_sources[(key, entity_type)].most_common(1)[0][0]
                entities.append(
                    {
                        "name": name,
                        "type": entity_type,
                        "count": count,
                        "source": source,
                        "canonical_id": stable_entity_id(name, entity_type),
                    }
                )

        def display_name(key: str, entity_type: str) -> str:
            candidates = Counter(
                {
                    name: count
                    for (name, candidate_type), count in self.entity_counts[key].items()
                    if candidate_type == entity_type
                }
            )
            if candidates:
                return candidates.most_common(1)[0][0]
            return self.entity_counts[key].most_common(1)[0][0][0]

        relations = []
        for key, count in self.fact_counts.items():
            head_key, relation, tail_key, head_type, tail_type = key
            head_variant = display_name(head_key, head_type)
            tail_variant = display_name(tail_key, tail_type)
            relations.append(
                {
                    "head": head_variant,
                    "head_type": head_type,
                    "relation": relation,
                    "tail": tail_variant,
                    "tail_type": tail_type,
                    "count": count,
                    "source": self.fact_sources[key].most_common(1)[0][0],
                }
            )
        benchmark_relations = []
        for key, count in self.benchmark_fact_counts.items():
            head_key, relation, tail_key, head_type, tail_type, predicate = key
            benchmark_relations.append(
                {
                    "head": display_name(head_key, head_type),
                    "head_type": head_type,
                    "relation": relation,
                    "predicate": predicate,
                    "tail": display_name(tail_key, tail_type),
                    "tail_type": tail_type,
                    "count": count,
                    "source": "CMeIE-V2_train",
                }
            )
        return {
            "format": "medigraph_fast_extractor",
            "version": "1.0.0",
            "method": "supervised_lexicon_trie_and_schema_relation_baseline",
            "claim": "deterministic non-neural CPU baseline",
            "source_checksums": dict(sorted(source_checksums.items())),
            "counts": {
                "entities": len(entities),
                "relations": len(relations),
                "cmeie_benchmark_relations": len(benchmark_relations),
            },
            "entities": sorted(entities, key=lambda item: (item["name"].casefold(), item["type"])),
            "relations": sorted(
                relations,
                key=lambda item: (item["head"].casefold(), item["relation"], item["tail"].casefold()),
            ),
            "cmeie_benchmark_relations": sorted(
                benchmark_relations,
                key=lambda item: (item["head"].casefold(), item["relation"], item["tail"].casefold()),
            ),
        }


def main() -> None:
    missing = [str(path) for path in (CMEIE_TRAIN, DIAKG_DIR, KG_GRAPH) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required source(s): " + ", ".join(missing))
    builder = ArtifactBuilder()
    print("=== Building reproducible L1 fast extractor ===")
    cmeie_records = builder.read_cmeie(CMEIE_TRAIN)
    diakg_documents = builder.read_diakg(DIAKG_DIR)
    graph_nodes, graph_edges, link_entries = builder.read_graph(KG_GRAPH)
    checksums = {
        "CMeIE-V2_train.jsonl": _sha256(CMEIE_TRAIN),
        "DIAKG_directory_manifest": hashlib.sha256(
            "".join(
                f"{path.name}:{_sha256(path)}\n"
                for path in sorted(DIAKG_DIR.glob("*.json"), key=lambda value: int(value.stem))
            ).encode("utf-8")
        ).hexdigest(),
        "cm3kg_graph.json": _sha256(KG_GRAPH),
    }
    artifact = builder.artifact(checksums)
    FAST_EXTRACTOR_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    FAST_EXTRACTOR_ARTIFACT.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    # Expand the linking KB beyond CM3KG: add the full medical term inventory
    # observed in CMeIE-V2 train and DIAKG so real mentions link at a much higher
    # rate. CM3KG entries keep priority (their canonical_id/aliases win on surface
    # conflicts because EntityLinker uses setdefault on the surface->id map).
    from medigraph.schema.normalize import canonical_key  # noqa: E402

    covered: set[str] = set()
    for entry in link_entries:
        covered.add(canonical_key(entry.name))
        for alias in entry.aliases:
            covered.add(canonical_key(alias))
    extra_entries: list[LinkEntry] = []
    for key, counter in builder.entity_counts.items():
        (name, entity_type), _ = counter.most_common(1)[0]
        ckey = canonical_key(name)
        if not ckey or ckey in covered:
            continue
        covered.add(ckey)
        source = builder.entity_sources.get((key, entity_type), Counter()).most_common(1)
        extra_entries.append(LinkEntry(
            canonical_id=stable_entity_id(name, entity_type),
            name=name, type=entity_type, aliases=[],
            source=source[0][0] if source else "CMeIE-V2_train",
        ))
    all_entries = link_entries + extra_entries
    EntityLinker(all_entries).save(ENTITY_LINKER_ARTIFACT)
    print(f"CMeIE records: {cmeie_records}; DIAKG documents: {diakg_documents}")
    print(f"CM3KG import: {graph_nodes} nodes / {graph_edges} edges")
    print(f"Artifact: {artifact['counts']} -> {FAST_EXTRACTOR_ARTIFACT}")
    print(f"Entity linker: {len(all_entries)} entries "
          f"(CM3KG {len(link_entries)} + CMeIE/DIAKG {len(extra_entries)}) -> {ENTITY_LINKER_ARTIFACT}")


if __name__ == "__main__":
    main()
