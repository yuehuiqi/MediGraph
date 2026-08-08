"""Analysis endpoints: graph-aware NL2SQL over the relational projection."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from service import deps
from service.cache import cache
from service.observability import record_llm_stats_delta
from service.schemas import NL2SQLRequest, NL2SQLResponse

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
log = logging.getLogger("medigraph.api.analysis")


@router.post(
    "/nl2sql",
    response_model=NL2SQLResponse,
    summary="自然语言转 SQL 并执行",
    description=(
        "图谱词表 Schema-Linking → 确定性快路径或 LLM 生成 → AST 只读校验 → "
        "SQLite authorizer 二次拦截 → 限时限行执行。"
        "generation_mode 标明本次是模板路由还是 LLM 生成。"
    ),
)
def nl2sql(payload: NL2SQLRequest) -> NL2SQLResponse:
    try:
        engine = deps.get_nl2sql()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    started = time.perf_counter()
    # Cache only pays here on the LLM-generated path (seconds); the deterministic
    # template path is already ~8 ms. Both are cached for uniformity -- the key
    # includes the engine backend, so sqlite/postgres results never cross.
    cache_inputs = {"question": payload.question, "backend": engine.backend}
    cached = cache.get("nl2sql", cache_inputs)
    if cached is not None:
        cached["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return NL2SQLResponse(**cached)

    llm = deps.get_llm()
    stats_before = llm.stats.summary()
    try:
        result = engine.query(payload.question)
    except Exception as exc:  # noqa: BLE001
        log.exception("nl2sql failed")
        raise HTTPException(status_code=502, detail=f"nl2sql failed: {exc}") from exc

    record_llm_stats_delta(stats_before, llm.stats.summary(), mode="blocking")
    elapsed_ms = (time.perf_counter() - started) * 1000
    # PostgreSQL aggregates (AVG/SUM over numeric) come back as Decimal, which
    # neither JSON caching nor strict clients handle; normalise at the boundary.
    from decimal import Decimal

    rows = [
        [float(value) if isinstance(value, Decimal) else value for value in row]
        for row in result.get("rows", [])
    ]
    log.info(
        "nl2sql",
        extra={
            "generation_mode": result.get("generation_mode"),
            "row_count": len(rows),
            "had_error": bool(result.get("error")),
            "duration_ms": round(elapsed_ms, 2),
        },
    )
    response = NL2SQLResponse(
        question=result.get("question", payload.question),
        sql=result.get("sql", ""),
        columns=list(result.get("columns", [])),
        rows=rows,
        row_count=len(rows),
        generation_mode=result.get("generation_mode", "unknown"),
        error=result.get("error", ""),
        elapsed_ms=round(elapsed_ms, 2),
    )
    if not response.error:  # never cache failures: they should retry, not persist
        cache.set("nl2sql", cache_inputs, response.model_dump())
    return response
