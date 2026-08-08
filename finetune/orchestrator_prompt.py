"""Shared definitions for the <1B data-processing orchestrator.

The SAME system prompt and operator catalog are used at data-synthesis, training,
and inference time -- consistency is what lets a 0.8B model learn the mapping
(natural-language goal -> operator DAG JSON). Canonical pipeline patterns provide
trustworthy gold labels for the synthesized training set.
"""
from __future__ import annotations

import json

# Operator toolbox shown to the model (mirrors medigraph/operators).
CATALOG = [
    {"op": "text_clean", "desc": "清洗文本：去网页噪声/页眉页脚/链接/短碎片", "args": {}},
    {"op": "chunker", "desc": "按标题与字数切块", "args": {"max_chars": "可选,整数"}},
    {"op": "medical_ner", "desc": "抽取医学实体(疾病/症状/药物/标志物/基因等)", "args": {}},
    {"op": "medical_re", "desc": "抽取实体间关系三元组", "args": {}},
    {"op": "triple_validator", "desc": "三元组校验(schema/去重/置信度/冲突)", "args": {"min_confidence": "可选,0-1"}},
]

SYSTEM = (
    "你是数据处理流程编排器。根据用户的自然语言需求，从给定算子里规划一个合法可执行的算子 DAG。\n"
    "可用算子：text_clean, chunker, medical_ner, medical_re, triple_validator。\n"
    "依赖规则：chunker 依赖 text_clean；medical_ner 依赖 chunker；medical_re 依赖 chunker 和 medical_ner；"
    "triple_validator 依赖 medical_re。只输出 JSON：{\"dag\":[{\"id\":\"n1\",\"op\":\"...\",\"args\":{},\"deps\":[]}]}。"
)


def build_user(goal: str) -> str:
    return f"需求：{goal}\n请输出算子 DAG（只输出 JSON）。"


def _node(i: int, op: str, deps: list[str], args: dict | None = None) -> dict:
    return {"id": f"n{i}", "op": op, "args": args or {}, "deps": deps}


def _pipeline(ops_with_args: list[tuple[str, dict]]) -> list[dict]:
    """Build a linear DAG honoring dependencies (medical_re also deps on ner)."""
    nodes, idx_of = [], {}
    for i, (op, args) in enumerate(ops_with_args, 1):
        deps = []
        if op == "chunker" and "text_clean" in idx_of:
            deps = [idx_of["text_clean"]]
        elif op == "medical_ner" and "chunker" in idx_of:
            deps = [idx_of["chunker"]]
        elif op == "medical_re":
            deps = [d for d in (idx_of.get("chunker"), idx_of.get("medical_ner")) if d]
        elif op == "triple_validator" and "medical_re" in idx_of:
            deps = [idx_of["medical_re"]]
        nid = f"n{i}"
        nodes.append(_node(i, op, deps, args))
        idx_of[op] = nid
    return nodes


# Canonical (intent -> gold DAG) patterns. Each spawns many paraphrased goals.
def patterns() -> list[dict]:
    P = []
    P.append({"intent": "只清洗文本去噪", "dag": _pipeline([("text_clean", {})])})
    P.append({"intent": "清洗后切块", "dag": _pipeline([("text_clean", {}), ("chunker", {})])})
    P.append({"intent": "清洗、切块、抽取医学实体", "dag": _pipeline([("text_clean", {}), ("chunker", {}), ("medical_ner", {})])})
    P.append({"intent": "清洗、切块、抽实体、抽关系三元组",
              "dag": _pipeline([("text_clean", {}), ("chunker", {}), ("medical_ner", {}), ("medical_re", {})])})
    P.append({"intent": "完整流水线:清洗->切块->实体->关系->校验",
              "dag": _pipeline([("text_clean", {}), ("chunker", {}), ("medical_ner", {}),
                                 ("medical_re", {}), ("triple_validator", {})])})
    # arg variants
    P.append({"intent": "清洗并按指定字数切块(800)",
              "dag": _pipeline([("text_clean", {}), ("chunker", {"max_chars": 800})])})
    P.append({"intent": "完整流水线且校验置信度阈值0.8",
              "dag": _pipeline([("text_clean", {}), ("chunker", {}), ("medical_ner", {}),
                                 ("medical_re", {}), ("triple_validator", {"min_confidence": 0.8})])})
    # already-clean input (skip text_clean)
    P.append({"intent": "文本已干净,只切块并抽实体", "dag": _pipeline([("chunker", {}), ("medical_ner", {})])})
    P.append({"intent": "只对文本抽取医学实体", "dag": [_node(1, "medical_ner", [])]})
    # build KG end-to-end (full pipeline, common phrasing)
    P.append({"intent": "从原始医疗文档构建知识图谱三元组",
              "dag": _pipeline([("text_clean", {}), ("chunker", {}), ("medical_ner", {}),
                                 ("medical_re", {}), ("triple_validator", {})])})
    # --- arg-value diversity: teach the model to extract max_chars / min_confidence ---
    P.append({"intent": "清洗并按500字切块", "dag": _pipeline([("text_clean", {}), ("chunker", {"max_chars": 500})])})
    P.append({"intent": "清洗并以每块1000字切分", "dag": _pipeline([("text_clean", {}), ("chunker", {"max_chars": 1000})])})
    P.append({"intent": "完整流水线,切块大小1500", "dag": _pipeline([("text_clean", {}), ("chunker", {"max_chars": 1500}),
              ("medical_ner", {}), ("medical_re", {}), ("triple_validator", {})])})
    P.append({"intent": "完整流水线,低置信阈值0.6过滤三元组", "dag": _pipeline([("text_clean", {}), ("chunker", {}),
              ("medical_ner", {}), ("medical_re", {}), ("triple_validator", {"min_confidence": 0.6})])})
    P.append({"intent": "完整流水线,高置信0.9严格校验", "dag": _pipeline([("text_clean", {}), ("chunker", {}),
              ("medical_ner", {}), ("medical_re", {}), ("triple_validator", {"min_confidence": 0.9})])})
    P.append({"intent": "完整流水线,块800且校验阈值0.85", "dag": _pipeline([("text_clean", {}), ("chunker", {"max_chars": 800}),
              ("medical_ner", {}), ("medical_re", {}), ("triple_validator", {"min_confidence": 0.85})])})
    # --- more partial pipelines (robustness) ---
    P.append({"intent": "文本已切好块,直接抽实体和关系", "dag": _pipeline([("medical_ner", {}), ("medical_re", {})])})
    P.append({"intent": "把已清洗文本按标题切块", "dag": [_node(1, "chunker", [])]})
    P.append({"intent": "清洗、切块、抽实体、抽关系并校验置信度0.7",
              "dag": _pipeline([("text_clean", {}), ("chunker", {}), ("medical_ner", {}),
                                 ("medical_re", {}), ("triple_validator", {"min_confidence": 0.7})])})
    return P


def canonical_signature(dag: list[dict]) -> tuple:
    """Order + dependency signature for exact-match DAG scoring.

    Dependencies are resolved from node-ids to the *ops* they point to, so the
    signature is independent of how ids are named.
    """
    id2op = {n.get("id"): n.get("op") for n in dag if isinstance(n, dict)}
    sig = []
    for n in dag:
        if not isinstance(n, dict) or "op" not in n:
            continue
        dep_ops = tuple(sorted(id2op.get(d, d) for d in n.get("deps", []) or []))
        sig.append((n["op"], dep_ops))
    return tuple(sig)


def dag_to_output(dag: list[dict]) -> str:
    return json.dumps({"dag": dag}, ensure_ascii=False)
