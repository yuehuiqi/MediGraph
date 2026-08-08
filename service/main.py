"""FastAPI application exposing MediGraph as an HTTP service.

    python -m uvicorn service.main:app --host 127.0.0.1 --port 8020
    # or: python service/main.py

This is a thin transport layer: every route delegates to the same operator and agent
implementations used by the CLI demos, the MCP tools and the DataMate packages. The
service adds what a long-lived process needs and the offline entry points do not --
cached heavy resources, readiness probes, request correlation, metrics and streaming.

Endpoints
    GET  /healthz              liveness: process is up
    GET  /readyz               readiness: per-dependency detail, 503 when degraded
    GET  /metrics              Prometheus exposition
    POST /api/v1/extract       entity + relation extraction
    POST /api/v1/kg/build      text -> knowledge graph
    POST /api/v1/kg/qa         graph QA (blocking)
    POST /api/v1/kg/qa/stream  graph QA (SSE)
    POST /api/v1/analysis/nl2sql
    GET  /docs                 OpenAPI UI
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow `python service/main.py` as well as `-m uvicorn service.main:app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from service import deps  # noqa: E402
from service.observability import (  # noqa: E402
    ObservabilityMiddleware,
    configure_logging,
    metrics_response,
    request_id_var,
)
from service.routers import analysis, extract, kg  # noqa: E402
from service.schemas import (  # noqa: E402
    ErrorResponse,
    HealthResponse,
    ReadyCheck,
    ReadyResponse,
)

SERVICE_NAME = "medigraph-api"
VERSION = "2.0.0"

log = logging.getLogger("medigraph.service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.getenv("MEDIGRAPH_LOG_LEVEL", "INFO"))
    log.info(
        "service starting",
        extra={"service": SERVICE_NAME, "version": VERSION},
    )
    # Deliberately no eager model loading: startup stays fast and a missing optional
    # artefact surfaces on /readyz instead of crash-looping the process.
    yield
    log.info("service stopping")


app = FastAPI(
    title="MediGraph Agent API",
    version=VERSION,
    description=(
        "医疗「数据—知识—洞察」智能体的 HTTP 接口。"
        "抽取、建图、图谱问答（含 SSE 流式）与图谱驱动 NL2SQL。"
    ),
    lifespan=lifespan,
)

app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    # Local demo/BI pages only. Widen deliberately if this is ever exposed.
    allow_origins=[
        origin
        for origin in os.getenv("MEDIGRAPH_CORS_ORIGINS", "http://127.0.0.1:8020,http://localhost:8020").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(extract.router)
app.include_router(kg.router)
app.include_router(analysis.router)


# --------------------------------------------------------------------------- #
# health / metrics
# --------------------------------------------------------------------------- #
@app.get("/healthz", response_model=HealthResponse, tags=["ops"], summary="存活探针")
def healthz() -> HealthResponse:
    """Liveness only: never touches dependencies, so a degraded DB cannot cause a
    restart loop under an orchestrator."""
    return HealthResponse(status="ok", service=SERVICE_NAME, version=VERSION)


@app.get("/readyz", response_model=ReadyResponse, tags=["ops"], summary="就绪探针")
def readyz() -> JSONResponse:
    """Readiness with per-dependency detail.

    Reports `degraded` (503) when any optional artefact is missing, but still lists
    every check so an operator can see *which* one -- more useful than a bare 503.
    """
    checks = deps.readiness_checks()
    ready = all(check["ready"] for check in checks)
    body = ReadyResponse(
        status="ready" if ready else "degraded",
        checks=[ReadyCheck(**check) for check in checks],
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content=body.model_dump(),
    )


@app.get("/metrics", tags=["ops"], summary="Prometheus 指标", include_in_schema=False)
def metrics():
    return metrics_response()


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a correlated error envelope instead of leaking a traceback.

    The stack trace goes to the structured log under the same request_id, so an
    operator can join the client-visible id to the server-side detail.
    """
    log.exception("unhandled error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail=type(exc).__name__,
            request_id=request_id_var.get(),
        ).model_dump(),
    )


def main() -> None:
    import uvicorn

    configure_logging(os.getenv("MEDIGRAPH_LOG_LEVEL", "INFO"))
    uvicorn.run(
        app,
        host=os.getenv("MEDIGRAPH_API_HOST", "127.0.0.1"),
        port=int(os.getenv("MEDIGRAPH_API_PORT", "8020")),
        log_config=None,  # keep our JSON handler
    )


if __name__ == "__main__":
    main()
