"""Extraction endpoint: text -> entities + triples via the L1/L2/L3 cascade.

Reuses the registered operator implementations rather than re-implementing the
cascade, so the HTTP surface, the MCP tools and the DAG all execute the same code.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from service import deps
from service.cache import cache
from service.observability import record_llm_stats_delta
from service.schemas import ExtractRequest, ExtractResponse

router = APIRouter(prefix="/api/v1", tags=["extraction"])
log = logging.getLogger("medigraph.api.extract")


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="医疗实体与关系抽取",
    description=(
        "对输入文本执行 L1 神经 GPLinker / L1' 词典 / L2 LLM 置信度路由抽取，"
        "可选 L3 实体链接与规范 ID。响应中的 routing 字段说明本次命中的层级。"
    ),
)
def extract(payload: ExtractRequest) -> ExtractResponse:
    deps.ensure_operators()
    from medigraph.operators.base import get_operator

    started = time.perf_counter()
    # Idempotent per (text, backend, linking) under the current model fingerprint,
    # so a repeat request skips ~7s of model forward passes. `elapsed_ms` is
    # rewritten to the cache-hit latency: reporting the original compute time for
    # a cached response would misrepresent what this request cost.
    cache_inputs = {
        "text": payload.text,
        "backend": payload.backend or "",
        "link": payload.link_entities,
    }
    cached = cache.get("extract", cache_inputs)
    if cached is not None:
        cached["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return ExtractResponse(**cached)

    llm = deps.get_llm()
    stats_before = llm.stats.summary()

    # `backend` is a constructor argument, not a run() input, so an override needs a
    # fresh operator. That is cheap: the underlying model/lexicon/linker loaders are
    # all lru_cached, so only the thin wrapper is rebuilt.
    if payload.backend:
        from medigraph.operators.medical_ner import MedicalNEROperator

        ner = MedicalNEROperator(llm=llm, backend=payload.backend)
    else:
        ner = get_operator("medical_ner")
    relation_extractor = get_operator("medical_re")

    try:
        ner_out = ner.run({"text": payload.text})
        entities = ner_out.get("entities", []) or []

        linking_report: dict = {}
        if payload.link_entities and entities:
            linked = get_operator("entity_linker").run({"entities": entities})
            entities = linked.get("entities", entities)
            linking_report = linked.get("linking_report", {})

        re_out = relation_extractor.run({"text": payload.text, "entities": entities})
        triples = re_out.get("triples", []) or []
    except Exception as exc:  # noqa: BLE001 - surfaced as 502, details in logs
        log.exception("extraction failed")
        raise HTTPException(status_code=502, detail=f"extraction failed: {exc}") from exc

    record_llm_stats_delta(stats_before, llm.stats.summary(), mode="blocking")
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "extracted",
        extra={
            "chars": len(payload.text),
            "entities": len(entities),
            "triples": len(triples),
            "routing_level": (ner_out.get("routing") or {}).get("level"),
            "duration_ms": round(elapsed_ms, 2),
        },
    )
    response = ExtractResponse(
        entities=entities,
        triples=triples,
        routing=ner_out.get("routing", {}) or {},
        linking_report=linking_report,
        elapsed_ms=round(elapsed_ms, 2),
    )
    cache.set("extract", cache_inputs, response.model_dump())
    return response
