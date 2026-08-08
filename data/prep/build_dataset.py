r"""Build the competition dataset suite from CM3KG + CMeIE-V2 + DiaKG (+ pathology).

Produces, under CCF/data/:
  kg/cm3kg_graph.json   -- large structured KG (disease-symptom/drug/check/dept/complication)
                           also copied to outputs/graph.json so QA/analysis/skill use it
  corpus/*.txt          -- raw medical texts for Task-1 ETL + Task-2 raw->graph extraction
  gold/ner_re_gold.json -- NER/RE gold (CMeIE-V2 dev + DiaKG, mapped to our ontology) for F1

Source datasets are expected one level above the repo root (../{CM3KG,CMeIE-V2,DIAKG}).

Usage:
  python data/prep/build_dataset.py
  python data/prep/build_dataset.py --max-diseases 1500 --n-gold-cmeie 200
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT  # noqa: E402
from medigraph.schema.ontology import RELATION_TYPES  # noqa: E402
from medigraph.schema.normalize import is_valid_entity_name  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from data.prep.ontology_map import (  # noqa: E402
    CM3KG_COLS, CMEIE_ENT, CMEIE_REL, CMEIE_REL_TAIL_TYPE, DIAKG_ENT, DIAKG_REL,
)

enable_utf8()
REPO = PROJECT_ROOT.parent
SRC_CM3KG = REPO / "CM3KG" / "medical.csv"
SRC_CMEIE_DEV = REPO / "CMeIE-V2" / "CMeIE-V2_dev.jsonl"
SRC_CMEIE_TRAIN = REPO / "CMeIE-V2" / "CMeIE-V2_train.jsonl"
SRC_DIAKG = REPO / "DIAKG" / "0521_new_format"
def _find_pathology_dir() -> Path:
    override = os.environ.get("PATHOLOGY_SOURCE_DIR", "").strip()
    if override:
        return Path(override)
    for path in REPO.rglob("webpath_output/markdown"):
        if path.is_dir():
            return path
    return REPO / "pathology" / "webpath_output" / "markdown"


SRC_PATHOLOGY = _find_pathology_dir()

CORPUS = DATA_DIR / "corpus"
GOLD = DATA_DIR / "gold"
KG = DATA_DIR / "kg"


def _as_list(val: str) -> list[str]:
    val = (val or "").strip()
    if not val:
        return []
    try:
        out = ast.literal_eval(val)
        if isinstance(out, (list, tuple)):
            return [str(x).strip() for x in out if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [v.strip() for v in re.split(r"[,，;；、]", val) if v.strip()]


# ----------------------------- CM3KG -> KG -------------------------------- #
def build_cm3kg_graph(max_diseases: int) -> dict:
    nodes: dict[str, str] = {}   # name -> type
    edges: list[dict] = []
    seen = set()
    csv.field_size_limit(10_000_000)
    with open(SRC_CM3KG, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            disease = (row.get("name") or "").strip()
            if not disease or not is_valid_entity_name(disease):
                continue
            nodes[disease] = "Disease"
            has_edge = False
            for col, (rel, tail_type) in CM3KG_COLS.items():
                for tail in _as_list(row.get(col, "")):
                    tail = tail.strip()
                    if not tail or tail == disease or not is_valid_entity_name(tail):
                        continue
                    nodes.setdefault(tail, tail_type)
                    key = (disease, rel, tail)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "head": disease, "head_type": "Disease", "relation": rel,
                        "relation_zh": RELATION_TYPES.get(rel, rel), "tail": tail,
                        "tail_type": tail_type, "confidence": 1.0, "source": "CM3KG",
                    })
                    has_edge = True
            if has_edge:
                n += 1
            if n >= max_diseases:
                break
    graph = {"nodes": [{"id": k, "type": v} for k, v in nodes.items()], "edges": edges}
    KG.mkdir(parents=True, exist_ok=True)
    (KG / "cm3kg_graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    # also make it the default downstream graph (QA/analysis/skill/A2A/MCP)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    print(f"[CM3KG] graph: {len(graph['nodes'])} nodes, {len(edges)} edges "
          f"-> data/kg/cm3kg_graph.json (+ outputs/graph.json)")
    return graph


# ------------------------- CM3KG -> in-domain gold ----------------------- #
# A *controlled* in-domain extraction benchmark: real CM3KG structured facts
# (authoritative ~8800-disease KG) rendered into natural clinical sentences, so
# every gold entity is verifiably present in the input text. Complements the
# hard, real-prose CMeIE/DiaKG public benchmark. Clearly labelled as templated.
_CM3KG_CLAUSE = {  # relation -> ([sentence template variants], max tails)
    "has_symptom": (["{d}的常见症状包括{items}。", "{d}患者多表现为{items}。",
                     "临床上，{d}可出现{items}等表现。", "{d}常伴有{items}。"], 4),
    "recommend_drug": (["治疗{d}可选用{items}等药物。", "{d}的常用药物有{items}。",
                        "针对{d}，临床可给予{items}。", "{d}可使用{items}进行治疗。"], 3),
    "need_examination": (["确诊{d}通常需要{items}等检查。", "{d}的诊断常借助{items}。",
                          "评估{d}时一般会做{items}。", "{d}需进行{items}以辅助诊断。"], 2),
    "complication": (["{d}可能并发{items}。", "{d}若控制不佳可引起{items}。",
                      "{d}的常见并发症为{items}。"], 2),
    "treated_in_department": (["{d}患者一般在{items}就诊。", "{d}通常归{items}诊治。",
                              "{d}多于{items}就诊。"], 1),
}
_CM3KG_ORDER = ["has_symptom", "recommend_drug", "need_examination", "complication", "treated_in_department"]


def build_cm3kg_gold(n_gold: int, seed: int = 42) -> list[dict]:
    if n_gold <= 0:
        return []
    rng = random.Random(seed)
    csv.field_size_limit(10_000_000)
    gold: list[dict] = []
    with open(SRC_CM3KG, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = (row.get("name") or "").strip()
            if not disease or not is_valid_entity_name(disease):
                continue
            # collect tails per relation from structured columns
            rel_tails: dict[str, list[str]] = {}
            for col, (rel, tail_type) in CM3KG_COLS.items():
                for tail in _as_list(row.get(col, "")):
                    tail = tail.strip()
                    if tail and tail != disease and len(tail) <= 12 and is_valid_entity_name(tail):
                        rel_tails.setdefault(rel, [])
                        if tail not in rel_tails[rel]:
                            rel_tails[rel].append(tail)
            present = [r for r in _CM3KG_ORDER if rel_tails.get(r)]
            if len(present) < 2:  # need a non-trivial sample
                continue
            text_parts, entities, triples = [], [{"name": disease, "type": "Disease"}], []
            seen_ent = {("Disease", disease)}
            rng.shuffle(present)  # vary clause order so text isn't templated-regular
            for rel in present:
                tmpls, cap = _CM3KG_CLAUSE[rel]
                tmpl = rng.choice(tmpls)
                tails = rel_tails[rel][:]
                rng.shuffle(tails)
                tails = tails[:cap]
                tail_type = next(tt for c, (r, tt) in CM3KG_COLS.items() if r == rel)
                text_parts.append(tmpl.format(d=disease, items="、".join(tails)))
                for t in tails:
                    if (tail_type, t) not in seen_ent:
                        seen_ent.add((tail_type, t))
                        entities.append({"name": t, "type": tail_type})
                    triples.append({"head": disease, "relation": rel, "tail": t})
            gold.append({"text": "".join(text_parts), "entities": entities,
                         "triples": triples, "source": "CM3KG"})
            if len(gold) >= n_gold:
                break
    out = {"description": "CM3KG-derived controlled in-domain NER/RE gold "
           "(real structured facts rendered into clinical sentences; every gold "
           "entity is present in the text).", "source": "CM3KG", "samples": gold}
    (GOLD / "cm3kg_gold.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[CM3KG-gold] {len(gold)} samples -> data/gold/cm3kg_gold.json")
    return gold


# ----------------- Pathology ontology-coverage probe gold ---------------- #
# A small *curated* probe set of real pathology/clinical facts whose sole job is
# to demonstrate that the FULL ontology (all 11 entity types / 14 relations) is
# populated and extractable — covering the pathology-domain types (Tumor /
# Biomarker / Gene / Morphology) that no structured public CN dataset provides.
# Clearly labelled as a coverage probe (not the headline benchmark). Every entity
# string appears verbatim in the text.
_PATH_PROBE = [
    ("胃肠间质瘤是一种间叶源性肿瘤，CD117 和 DOG1 多呈阳性，S-100 通常阴性，常与 KIT 基因突变相关，镜下多为梭形细胞，好发于胃壁。",
     [("胃肠间质瘤","Tumor"),("CD117","Biomarker"),("DOG1","Biomarker"),("S-100","Biomarker"),("KIT","Gene"),("梭形细胞","Morphology"),("胃壁","Body")],
     [("胃肠间质瘤","positive_marker","CD117"),("胃肠间质瘤","positive_marker","DOG1"),("胃肠间质瘤","negative_marker","S-100"),("胃肠间质瘤","associated_gene","KIT"),("胃肠间质瘤","has_morphology","梭形细胞"),("胃肠间质瘤","located_in","胃壁")]),
    ("神经母细胞瘤好发于肾上腺髓质，NSE 与 Syn 常呈阳性，多伴 MYCN 基因扩增，镜下可见 Homer-Wright 菊形团。",
     [("神经母细胞瘤","Tumor"),("肾上腺髓质","Body"),("NSE","Biomarker"),("Syn","Biomarker"),("MYCN","Gene"),("菊形团","Morphology")],
     [("神经母细胞瘤","located_in","肾上腺髓质"),("神经母细胞瘤","positive_marker","NSE"),("神经母细胞瘤","positive_marker","Syn"),("神经母细胞瘤","associated_gene","MYCN"),("神经母细胞瘤","has_morphology","菊形团")]),
    ("弥漫大B细胞淋巴瘤是非霍奇金淋巴瘤的一种亚型，CD20 呈阳性，CD3 阴性，常用利妥昔单抗治疗。",
     [("弥漫大B细胞淋巴瘤","Tumor"),("非霍奇金淋巴瘤","Tumor"),("CD20","Biomarker"),("CD3","Biomarker"),("利妥昔单抗","Drug")],
     [("弥漫大B细胞淋巴瘤","subtype_of","非霍奇金淋巴瘤"),("弥漫大B细胞淋巴瘤","positive_marker","CD20"),("弥漫大B细胞淋巴瘤","negative_marker","CD3"),("弥漫大B细胞淋巴瘤","recommend_drug","利妥昔单抗")]),
    ("二甲双胍禁用于严重肾功能不全患者；顺铂可引起恶心和耳鸣等不良反应。",
     [("二甲双胍","Drug"),("肾功能不全","Disease"),("顺铂","Drug"),("恶心","Symptom"),("耳鸣","Symptom")],
     [("二甲双胍","contraindication","肾功能不全"),("顺铂","adverse_reaction","恶心"),("顺铂","adverse_reaction","耳鸣")]),
    ("急性阑尾炎常需行阑尾切除术，患者多在普外科就诊，可并发腹膜炎。",
     [("急性阑尾炎","Disease"),("阑尾切除术","Procedure"),("普外科","Department"),("腹膜炎","Disease")],
     [("急性阑尾炎","treated_by_procedure","阑尾切除术"),("急性阑尾炎","treated_in_department","普外科"),("急性阑尾炎","complication","腹膜炎")]),
    ("急性心肌梗死的典型症状为胸痛和大汗，需检查心电图和肌钙蛋白，可使用阿司匹林。",
     [("急性心肌梗死","Disease"),("胸痛","Symptom"),("大汗","Symptom"),("心电图","Examination"),("肌钙蛋白","Examination"),("阿司匹林","Drug")],
     [("急性心肌梗死","has_symptom","胸痛"),("急性心肌梗死","has_symptom","大汗"),("急性心肌梗死","need_examination","心电图"),("急性心肌梗死","need_examination","肌钙蛋白"),("急性心肌梗死","recommend_drug","阿司匹林")]),
    ("肝细胞癌位于肝脏，AFP 常升高呈阳性，可与 TP53 基因突变相关，早期可行肝部分切除术，常在肝胆外科就诊。",
     [("肝细胞癌","Tumor"),("肝脏","Body"),("AFP","Biomarker"),("TP53","Gene"),("肝部分切除术","Procedure"),("肝胆外科","Department")],
     [("肝细胞癌","located_in","肝脏"),("肝细胞癌","positive_marker","AFP"),("肝细胞癌","associated_gene","TP53"),("肝细胞癌","treated_by_procedure","肝部分切除术"),("肝细胞癌","treated_in_department","肝胆外科")]),
    ("乳腺癌中 HER2 阳性者可用曲妥珠单抗治疗，ER 阴性提示预后较差，镜下可见浸润性导管结构，好发于乳腺。",
     [("乳腺癌","Tumor"),("HER2","Biomarker"),("曲妥珠单抗","Drug"),("ER","Biomarker"),("浸润性导管","Morphology"),("乳腺","Body")],
     [("乳腺癌","positive_marker","HER2"),("乳腺癌","recommend_drug","曲妥珠单抗"),("乳腺癌","negative_marker","ER"),("乳腺癌","has_morphology","浸润性导管"),("乳腺癌","located_in","乳腺")]),
    ("2型糖尿病是糖尿病的常见亚型，可并发糖尿病视网膜病变，患者多在内分泌科就诊，需检测糖化血红蛋白。",
     [("2型糖尿病","Disease"),("糖尿病","Disease"),("糖尿病视网膜病变","Disease"),("内分泌科","Department"),("糖化血红蛋白","Examination")],
     [("2型糖尿病","subtype_of","糖尿病"),("2型糖尿病","complication","糖尿病视网膜病变"),("2型糖尿病","treated_in_department","内分泌科"),("2型糖尿病","need_examination","糖化血红蛋白")]),
    ("胃腺癌好发于胃窦，CEA 可呈阳性，HER2 部分阳性，镜下可见印戒细胞，与 CDH1 基因相关。",
     [("胃腺癌","Tumor"),("胃窦","Body"),("CEA","Biomarker"),("HER2","Biomarker"),("印戒细胞","Morphology"),("CDH1","Gene")],
     [("胃腺癌","located_in","胃窦"),("胃腺癌","positive_marker","CEA"),("胃腺癌","positive_marker","HER2"),("胃腺癌","has_morphology","印戒细胞"),("胃腺癌","associated_gene","CDH1")]),
    ("糖皮质激素长期使用可引起骨质疏松和血糖升高，禁用于活动性消化性溃疡患者。",
     [("糖皮质激素","Drug"),("骨质疏松","Disease"),("血糖升高","Symptom"),("消化性溃疡","Disease")],
     [("糖皮质激素","adverse_reaction","骨质疏松"),("糖皮质激素","adverse_reaction","血糖升高"),("糖皮质激素","contraindication","消化性溃疡")]),
    ("急性白血病常采用化疗，患者可出现发热和贫血，需检查血常规。",
     [("急性白血病","Disease"),("化疗","Procedure"),("发热","Symptom"),("贫血","Symptom"),("血常规","Examination")],
     [("急性白血病","treated_by_procedure","化疗"),("急性白血病","has_symptom","发热"),("急性白血病","has_symptom","贫血"),("急性白血病","need_examination","血常规")]),
    ("肺腺癌位于肺，TTF-1 常呈阳性，与 EGFR 基因突变相关，相应患者可用吉非替尼治疗。",
     [("肺腺癌","Tumor"),("肺","Body"),("TTF-1","Biomarker"),("EGFR","Gene"),("吉非替尼","Drug")],
     [("肺腺癌","located_in","肺"),("肺腺癌","positive_marker","TTF-1"),("肺腺癌","associated_gene","EGFR"),("肺腺癌","recommend_drug","吉非替尼")]),
    ("前列腺癌可使 PSA 升高呈阳性，需行前列腺穿刺活检，患者多在泌尿外科就诊。",
     [("前列腺癌","Tumor"),("PSA","Biomarker"),("前列腺穿刺活检","Examination"),("泌尿外科","Department")],
     [("前列腺癌","positive_marker","PSA"),("前列腺癌","need_examination","前列腺穿刺活检"),("前列腺癌","treated_in_department","泌尿外科")]),
    ("恶性黑色素瘤好发于皮肤，S-100 与 HMB-45 呈阳性，镜下可见色素颗粒，可并发淋巴结转移。",
     [("恶性黑色素瘤","Tumor"),("皮肤","Body"),("S-100","Biomarker"),("HMB-45","Biomarker"),("色素颗粒","Morphology"),("淋巴结转移","Disease")],
     [("恶性黑色素瘤","located_in","皮肤"),("恶性黑色素瘤","positive_marker","S-100"),("恶性黑色素瘤","positive_marker","HMB-45"),("恶性黑色素瘤","has_morphology","色素颗粒"),("恶性黑色素瘤","complication","淋巴结转移")]),
    ("甲状腺乳头状癌是甲状腺癌的一种亚型，TG 与 TTF-1 阳性，常与 BRAF 基因突变相关，镜下可见毛玻璃样核。",
     [("甲状腺乳头状癌","Tumor"),("甲状腺癌","Tumor"),("TG","Biomarker"),("TTF-1","Biomarker"),("BRAF","Gene"),("毛玻璃样核","Morphology")],
     [("甲状腺乳头状癌","subtype_of","甲状腺癌"),("甲状腺乳头状癌","positive_marker","TG"),("甲状腺乳头状癌","positive_marker","TTF-1"),("甲状腺乳头状癌","associated_gene","BRAF"),("甲状腺乳头状癌","has_morphology","毛玻璃样核")]),
]


def build_pathology_probe() -> list[dict]:
    samples = []
    for text, ents, tris in _PATH_PROBE:
        # sanity: every entity/triple-arg must be present in the text
        for nm, _ in ents:
            assert nm in text, f"entity not in text: {nm}"
        samples.append({
            "text": text,
            "entities": [{"name": n, "type": t} for n, t in ents],
            "triples": [{"head": h, "relation": r, "tail": t} for h, r, t in tris],
            "source": "PathologyProbe",
        })
    out = {"description": "Curated ontology-coverage probe (pathology/clinical facts) "
           "demonstrating ALL 11 entity types and 14 relations are populated & "
           "extractable; entities verbatim in text. Coverage probe, not headline benchmark.",
           "source": "PathologyProbe", "samples": samples}
    (GOLD / "pathology_probe_gold.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[Pathology-probe] {len(samples)} samples -> data/gold/pathology_probe_gold.json")
    return samples


def build_showcase_graph(cm3kg_graph: dict, probe: list[dict]) -> dict:
    """A demonstration KG that exercises the FULL ontology: the CM3KG disease graph
    (5 relations) + the pathology-probe triples (the pathology relations/types). So
    the visible graph shows all 11 entity types & 14 relations, not just 5/5."""
    nodes = {n["id"]: n["type"] for n in cm3kg_graph["nodes"]}
    edges = list(cm3kg_graph["edges"])
    seen = {(e["head"], e["relation"], e["tail"]) for e in edges}
    for s in probe:
        for e in s["entities"]:
            nodes.setdefault(e["name"], e["type"])
        for t in s["triples"]:
            key = (t["head"], t["relation"], t["tail"])
            if key in seen:
                continue
            seen.add(key)
            edges.append({"head": t["head"], "head_type": "", "relation": t["relation"],
                          "relation_zh": RELATION_TYPES.get(t["relation"], t["relation"]),
                          "tail": t["tail"], "tail_type": "", "confidence": 1.0, "source": "PathologyProbe"})
    graph = {"nodes": [{"id": k, "type": v} for k, v in nodes.items()], "edges": edges}
    (KG / "ontology_showcase_graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    ntypes = len({v for v in nodes.values()}); rtypes = len({e["relation"] for e in edges})
    print(f"[Showcase] graph: {len(nodes)} nodes / {len(edges)} edges, "
          f"{ntypes} entity types / {rtypes} relations -> data/kg/ontology_showcase_graph.json")
    return graph


# ----------------------------- CMeIE-V2 ---------------------------------- #
def _cmeie_sample(rec: dict) -> dict | None:
    text = rec.get("text", "").strip()
    ents: dict[tuple, dict] = {}
    triples = []
    for spo in rec.get("spo_list", []):
        pred = spo.get("predicate", "")
        rel = CMEIE_REL.get(pred)
        if not rel:
            continue
        subj = str(spo.get("subject", "")).strip()
        obj_raw = spo.get("object", {})
        obj = str(obj_raw.get("@value", obj_raw) if isinstance(obj_raw, dict) else obj_raw).strip()
        styp = CMEIE_ENT.get(spo.get("subject_type", ""), "Disease")
        otyp_src = spo.get("object_type", {})
        otyp_src = otyp_src.get("@value") if isinstance(otyp_src, dict) else otyp_src
        otyp = CMEIE_ENT.get(otyp_src) or CMEIE_REL_TAIL_TYPE.get(rel) or "Symptom"
        if not subj or not obj:
            continue
        ents[(styp, subj)] = {"name": subj, "type": styp}
        ents[(otyp, obj)] = {"name": obj, "type": otyp}
        triples.append({"head": subj, "relation": rel, "tail": obj})
    if not triples:
        return None
    return {"text": text, "entities": list(ents.values()), "triples": triples, "source": "CMeIE-V2"}


def build_cmeie(n_gold: int, n_corpus: int) -> list[dict]:
    gold, corpus_texts = [], []
    with open(SRC_CMEIE_DEV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = _cmeie_sample(rec)
            if s:
                gold.append(s)
            if len(gold) >= n_gold:
                break
    # corpus from train texts (raw, for extraction demo / ETL)
    with open(SRC_CMEIE_TRAIN, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(corpus_texts) >= n_corpus:
                break
            try:
                t = json.loads(line).get("text", "").strip()
            except json.JSONDecodeError:
                continue
            if len(t) > 60:
                corpus_texts.append(t)
    for i, t in enumerate(corpus_texts):
        (CORPUS / f"cmeie_{i:03d}.txt").write_text(t, encoding="utf-8")
    print(f"[CMeIE] gold samples: {len(gold)} | corpus txts: {len(corpus_texts)}")
    return gold


# ----------------------------- DiaKG ------------------------------------- #
def build_diakg(n_gold: int, n_corpus_docs: int) -> list[dict]:
    gold = []
    files = sorted(SRC_DIAKG.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    n_docs = 0
    for fp in files:
        doc = json.loads(fp.read_text(encoding="utf-8"))
        doc_text_parts = []
        for para in doc.get("paragraphs", []):
            for sent in para.get("sentences", []):
                text = sent.get("sentence", "").strip()
                doc_text_parts.append(text)
                id2ent = {e["entity_id"]: e for e in sent.get("entities", [])}
                ents: dict[tuple, dict] = {}
                triples = []
                for r in sent.get("relations", []):
                    rel = DIAKG_REL.get(r.get("relation_type", ""))
                    if not rel:
                        continue
                    head = id2ent.get(r.get("head_entity_id"))   # e.g. Drug
                    tail = id2ent.get(r.get("tail_entity_id"))   # Disease
                    if not head or not tail:
                        continue
                    h_t = DIAKG_ENT.get(head.get("entity_type"))
                    t_t = DIAKG_ENT.get(tail.get("entity_type"))
                    if not h_t or not t_t:
                        continue
                    # canonicalize: Disease(tail) is subject of our relation
                    subj, subj_t = tail["entity"], t_t
                    obj, obj_t = head["entity"], h_t
                    ents[(subj_t, subj)] = {"name": subj, "type": subj_t}
                    ents[(obj_t, obj)] = {"name": obj, "type": obj_t}
                    triples.append({"head": subj, "relation": rel, "tail": obj})
                if triples and len(gold) < n_gold:
                    gold.append({"text": text, "entities": list(ents.values()), "triples": triples, "source": "DiaKG"})
        if n_docs < n_corpus_docs and doc_text_parts:
            (CORPUS / f"diakg_{fp.stem}.txt").write_text("\n".join(doc_text_parts), encoding="utf-8")
            n_docs += 1
    print(f"[DiaKG] gold samples: {len(gold)} | corpus docs: {n_docs}")
    return gold


# ----------------------------- pathology --------------------------------- #
def copy_pathology(n: int) -> int:
    if not SRC_PATHOLOGY.exists():
        return 0
    cleaned = sorted(SRC_PATHOLOGY.rglob("*_cleaned.md"))[:n]
    for p in cleaned:
        (CORPUS / f"pathology_{p.stem}.txt").write_text(
            p.read_text(encoding="utf-8-sig"), encoding="utf-8")
    print(f"[Pathology] corpus docs: {len(cleaned)}")
    return len(cleaned)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-diseases", type=int, default=1200, help="CM3KG diseases to load into KG")
    ap.add_argument("--n-gold-cmeie", type=int, default=150)
    ap.add_argument("--n-gold-diakg", type=int, default=100)
    ap.add_argument("--n-gold-cm3kg", type=int, default=120, help="CM3KG controlled in-domain gold size")
    ap.add_argument("--n-corpus-cmeie", type=int, default=40)
    ap.add_argument("--n-corpus-diakg", type=int, default=15)
    ap.add_argument("--n-pathology", type=int, default=5)
    args = ap.parse_args()

    for d in (CORPUS, GOLD, KG):
        d.mkdir(parents=True, exist_ok=True)

    print("=== Building MediGraph competition dataset suite ===")
    cm3kg_graph = build_cm3kg_graph(args.max_diseases)
    build_cm3kg_gold(args.n_gold_cm3kg)
    probe = build_pathology_probe()
    build_showcase_graph(cm3kg_graph, probe)
    gold = []
    gold += build_cmeie(args.n_gold_cmeie, args.n_corpus_cmeie)
    gold += build_diakg(args.n_gold_diakg, args.n_corpus_diakg)
    copy_pathology(args.n_pathology)

    report = {
        "description": "NER/RE gold standard mapped to the MediGraph ontology, "
                       "from CMeIE-V2 dev + DiaKG. Matching is normalization-aware.",
        "sources": ["CMeIE-V2_dev", "DiaKG"],
        "samples": gold,
    }
    (GOLD / "ner_re_gold.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    n_corpus = len(list(CORPUS.glob("*.txt")))
    print(f"\n[Gold] {len(gold)} samples -> data/gold/ner_re_gold.json")
    print(f"[Corpus] {n_corpus} txt files -> data/corpus/")
    print("Done. Downstream graph = CM3KG (outputs/graph.json).")


if __name__ == "__main__":
    main()
