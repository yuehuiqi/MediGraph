"""Knowledge-graph endpoints: build a graph, and answer questions over one.

`/kg/qa` has two shapes over the same retrieval path:
  * `POST /kg/qa`        -- blocking JSON, full evidence payload.
  * `POST /kg/qa/stream` -- SSE, prose streamed as generated, evidence on `done`.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from service import deps
from service.observability import record_llm_stats_delta
from service.schemas import (
    KGBuildRequest,
    KGBuildResponse,
    KGQARequest,
    KGQAResponse,
)
from service.stream import SSE_HEADERS, parse_last_event_id, sse_llm_stream

router = APIRouter(prefix="/api/v1/kg", tags=["knowledge-graph"])
log = logging.getLogger("medigraph.api.kg")


def _qa_agent(graph_name: str, hops: int):
    from medigraph.agents.qa_agent import QAAgent

    try:
        store = deps.get_graph(graph_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    deps.ensure_operators()
    return QAAgent(llm=deps.get_llm(), store=store, hops=hops)


@router.post(
    "/build",
    response_model=KGBuildResponse,
    summary="从文本构建知识图谱",
    description="切块 → 实体抽取 → 关系抽取 → 三元组校验，落盘为 outputs/ 下的图谱文件。",
)
def build(payload: KGBuildRequest) -> KGBuildResponse:
    from medigraph.agents.kg_agent import KGGenAgent

    try:
        target = deps.resolve_output_path(payload.graph_name, ".json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    deps.ensure_operators()
    llm = deps.get_llm()
    stats_before = llm.stats.summary()
    started = time.perf_counter()
    try:
        agent = KGGenAgent(llm=llm)
        build_stats = agent.build(
            [{"fileName": "api_request.txt", "text": payload.text}], verbose=False
        )
        validation = agent.store.audit() if hasattr(agent.store, "audit") else {}
        validation = {**(validation or {}), "build": build_stats or {}}
        agent.store.export_json(target)
    except Exception as exc:  # noqa: BLE001
        log.exception("kg build failed")
        raise HTTPException(status_code=502, detail=f"kg build failed: {exc}") from exc

    record_llm_stats_delta(stats_before, llm.stats.summary(), mode="blocking")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return KGBuildResponse(
        graph_name=target.name,
        num_entities=agent.store.g.number_of_nodes(),
        num_triples=agent.store.g.number_of_edges(),
        validation=validation or {},
        elapsed_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/qa",
    response_model=KGQAResponse,
    summary="图谱问答（阻塞）",
    description="多跳子图检索 + 逐边溯源；证据不足时按医疗安全策略拒答。",
)
def qa(payload: KGQARequest) -> KGQAResponse:
    agent = _qa_agent(payload.graph_name, payload.hops)
    llm = deps.get_llm()
    stats_before = llm.stats.summary()
    started = time.perf_counter()
    try:
        result = agent.answer(payload.question)
    except Exception as exc:  # noqa: BLE001
        log.exception("kg qa failed")
        raise HTTPException(status_code=502, detail=f"kg qa failed: {exc}") from exc

    record_llm_stats_delta(stats_before, llm.stats.summary(), mode="blocking")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return KGQAResponse(
        question=result["question"],
        answer=result["answer"],
        hops=payload.hops,
        confidence=result.get("answer_confidence", {}),
        evidence=result.get("citations", []),
        elapsed_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/qa/stream",
    summary="图谱问答（SSE 流式）",
    description=(
        "与 /kg/qa 相同的检索与证据链，但答案文本按 token 推送，首字延迟显著低于阻塞式。"
        "事件序列：meta → delta* → done|error。断线后带 Last-Event-ID 与 stream_id 重连可续传。"
    ),
    response_class=StreamingResponse,
)
async def qa_stream(
    payload: KGQARequest,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    stream_id: str | None = Header(default=None, alias="X-Stream-ID"),
) -> StreamingResponse:
    agent = _qa_agent(payload.graph_name, payload.hops)
    request_id = getattr(request.state, "request_id", "-")

    def produce(emit):
        # Retrieval happens inside answer(); only the composition step streams, so
        # the SSE client sees `meta` immediately, then a pause while the subgraph is
        # gathered, then prose.
        result = agent.answer(payload.question, on_token=emit)
        # The refusal paths (no evidence / confidence below threshold) return before
        # the composition step, so nothing was streamed. Emit their text too, so the
        # client's delta handling is uniform and a refusal is not an empty stream.
        if isinstance(result, dict) and result.get("refused") and result.get("answer"):
            emit(result["answer"])
        return result

    def done_extra(result) -> dict:
        if not isinstance(result, dict):
            return {}
        return {
            "refused": result.get("refused", False),
            "confidence": result.get("answer_confidence", {}),
            "evidence": result.get("citations", []),
            "retrieval_mode": result.get("retrieval_mode"),
            "evidence_used": result.get("evidence_used", 0),
        }

    body = sse_llm_stream(
        produce,
        request_id=request_id,
        last_event_id=parse_last_event_id(last_event_id),
        stream_id=stream_id,
        done_extra=done_extra,
    )
    return StreamingResponse(body, media_type="text/event-stream", headers=SSE_HEADERS)
