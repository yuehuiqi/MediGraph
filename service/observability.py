"""Structured logging, request correlation and Prometheus metrics.

Three concerns, deliberately in one small module because they share the request
context:

1. **request_id propagation** -- a ``ContextVar`` set by the middleware and picked
   up by the log formatter, so every line emitted while handling a request carries
   the same id without threading it through business-layer signatures.
2. **JSON logs** -- one object per line, so the output is greppable by field
   instead of by regex over prose.
3. **Prometheus metrics** -- request counters/histograms plus LLM-specific series
   (TTFT is tracked separately from total latency because streaming only improves
   the former).

The metric names follow Prometheus conventions: ``_total`` for counters,
``_seconds`` for duration histograms, base units (no milliseconds).
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pythonjsonlogger import json as jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

#: Correlation id for the in-flight request; "-" when running outside a request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "X-Request-ID"

# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
HTTP_REQUESTS = Counter(
    "medigraph_http_requests_total",
    "HTTP requests handled, by route/method/status.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "medigraph_http_request_duration_seconds",
    "End-to-end HTTP handler latency.",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
HTTP_IN_FLIGHT = Gauge(
    "medigraph_http_in_flight_requests",
    "Requests currently being handled.",
)
LLM_LATENCY = Histogram(
    "medigraph_llm_call_duration_seconds",
    "LLM call latency (full completion).",
    ["mode"],  # blocking | stream
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
)
LLM_TTFT = Histogram(
    "medigraph_llm_time_to_first_token_seconds",
    "Time to first streamed token. The metric streaming actually improves.",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0),
)
LLM_TOKENS = Counter(
    "medigraph_llm_tokens_total",
    "Tokens consumed/produced by LLM calls.",
    ["kind"],  # prompt | completion
)
LLM_ERRORS = Counter(
    "medigraph_llm_errors_total",
    "LLM calls that ultimately failed.",
)
# Populated from P2.3 onwards; declared here so /metrics has a stable shape.
CACHE_EVENTS = Counter(
    "medigraph_cache_events_total",
    "Cache lookups by outcome.",
    ["cache", "outcome"],  # outcome: hit | miss
)


def metrics_response() -> Response:
    """Prometheus scrape endpoint payload."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class _RequestIdFilter(logging.Filter):
    """Inject the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install a single JSON-lines handler on the root logger (idempotent)."""
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_medigraph_json", False):
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
            timestamp=False,
        )
    )
    handler.addFilter(_RequestIdFilter())
    handler._medigraph_json = True  # type: ignore[attr-defined]
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn installs its own colourised handlers; route them through ours so the
    # whole process emits one machine-readable stream.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


# --------------------------------------------------------------------------- #
# middleware
# --------------------------------------------------------------------------- #
class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request id, log one line per request, record metrics."""

    def __init__(self, app, logger_name: str = "medigraph.access"):
        super().__init__(app)
        self.log = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):
        # Honour an inbound id so a trace survives across service hops.
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        request.state.request_id = request_id

        # Use the matched route template, not the raw path: `/api/v1/kg/qa` stays a
        # single series instead of exploding cardinality on path parameters.
        route = request.scope.get("route")
        route_label = getattr(route, "path", None) or request.url.path

        started = time.perf_counter()
        status = 500
        HTTP_IN_FLIGHT.inc()
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            HTTP_IN_FLIGHT.dec()
            # Re-read the route: it is only populated after routing has run.
            route_now = request.scope.get("route")
            route_label = getattr(route_now, "path", None) or route_label
            HTTP_LATENCY.labels(request.method, route_label).observe(elapsed)
            HTTP_REQUESTS.labels(request.method, route_label, str(status)).inc()
            self.log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "route": route_label,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                    "client": request.client.host if request.client else None,
                },
            )
            request_id_var.reset(token)


def record_llm_stats_delta(before: dict, after: dict, mode: str) -> None:
    """Fold the delta between two `CallStats.summary()` snapshots into metrics.

    `LLMClient` already accumulates latency/token counters, so the service reads
    those instead of wrapping every call site.
    """
    calls = after.get("calls", 0) - before.get("calls", 0)
    if calls <= 0:
        return
    seconds = after.get("total_seconds", 0.0) - before.get("total_seconds", 0.0)
    LLM_LATENCY.labels(mode).observe(max(seconds, 0.0) / calls)
    for kind in ("prompt", "completion"):
        delta = after.get(f"{kind}_tokens", 0) - before.get(f"{kind}_tokens", 0)
        if delta > 0:
            LLM_TOKENS.labels(kind).inc(delta)
