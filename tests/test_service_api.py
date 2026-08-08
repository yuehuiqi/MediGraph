"""Offline tests for the HTTP service: SSE protocol, resume, observability, ops.

No network: the LLM client is replaced with a deterministic fake, so these run in
CI without an API key.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from service.stream import (  # noqa: E402
    MAX_BUFFERED_CHARS,
    StreamBuffer,
    StreamRegistry,
    parse_last_event_id,
    sse_comment,
    sse_frame,
)


# --------------------------------------------------------------------------- #
# SSE encoding
# --------------------------------------------------------------------------- #
def test_sse_frame_shape():
    frame = sse_frame("delta", {"text": "hi"}, event_id=3)
    assert frame == 'id: 3\nevent: delta\ndata: {"text": "hi"}\n\n'


def test_sse_frame_without_id_omits_id_line():
    assert "id:" not in sse_frame("meta", {"a": 1})


def test_sse_frame_keeps_payload_on_one_line():
    """A raw newline in `data:` would be parsed as a field break by the SSE spec."""
    frame = sse_frame("delta", {"text": "line1\nline2\r\nline3"})
    body = frame.split("data: ", 1)[1]
    assert body.count("\n") == 2  # only the two frame-terminating newlines
    assert json.loads(body.strip())["text"] == "line1\nline2\r\nline3"


def test_sse_frame_preserves_unicode():
    frame = sse_frame("delta", {"text": "高血压"})
    assert "高血压" in frame
    assert json.loads(frame.split("data: ", 1)[1].strip())["text"] == "高血压"


def test_sse_comment_is_a_comment():
    assert sse_comment("ka").startswith(":")


# --------------------------------------------------------------------------- #
# resume bookkeeping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, 0), ("", 0), ("5", 5), (" 7 ", 7), ("-3", 0), ("abc", 0), ("1.5", 0)],
)
def test_parse_last_event_id(header, expected):
    assert parse_last_event_id(header) == expected


def test_stream_buffer_sequence_numbers_are_one_based():
    buffer = StreamBuffer("s1")
    assert buffer.append("a") == 1
    assert buffer.append("b") == 2
    assert buffer.text == "ab"


def test_stream_buffer_replays_only_the_tail():
    buffer = StreamBuffer("s1")
    for piece in "abcd":
        buffer.append(piece)
    assert buffer.after(2) == [(3, "c"), (4, "d")]
    assert buffer.after(0) == [(1, "a"), (2, "b"), (3, "c"), (4, "d")]
    assert buffer.after(4) == []


def test_stream_buffer_marks_truncation_past_budget():
    buffer = StreamBuffer("s1")
    buffer.append("x" * (MAX_BUFFERED_CHARS + 1))
    assert buffer.truncated


def test_stream_registry_evicts_oldest():
    registry = StreamRegistry(max_streams=2)
    first = registry.create()
    second = registry.create()
    third = registry.create()
    assert registry.get(first.stream_id) is None  # evicted
    assert registry.get(second.stream_id) is second
    assert registry.get(third.stream_id) is third


def test_stream_registry_get_refreshes_recency():
    registry = StreamRegistry(max_streams=2)
    first = registry.create()
    second = registry.create()
    registry.get(first.stream_id)  # first becomes most-recent
    third = registry.create()
    assert registry.get(first.stream_id) is first
    assert registry.get(second.stream_id) is None


# --------------------------------------------------------------------------- #
# app-level: ops endpoints and observability
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    from service.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_healthz_does_not_touch_dependencies(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "medigraph-api"


def test_readyz_lists_every_check(client):
    response = client.get("/readyz")
    assert response.status_code in (200, 503)
    body = response.json()
    assert body["status"] in ("ready", "degraded")
    names = {check["name"] for check in body["checks"]}
    assert {"llm_api_key", "analytics_db"} <= names
    # Degraded must still enumerate which dependency failed.
    if body["status"] == "degraded":
        assert any(not check["ready"] for check in body["checks"])


def test_metrics_exposes_declared_series(client):
    client.get("/healthz")  # generate at least one observation
    text = client.get("/metrics").text
    for series in (
        "medigraph_http_requests_total",
        "medigraph_http_request_duration_seconds",
        "medigraph_llm_time_to_first_token_seconds",
    ):
        assert series in text


def test_request_id_is_returned_and_echoed(client):
    generated = client.get("/healthz").headers.get("X-Request-ID")
    assert generated and len(generated) == 16
    supplied = client.get("/healthz", headers={"X-Request-ID": "trace-abc"})
    assert supplied.headers["X-Request-ID"] == "trace-abc"


def test_metrics_label_uses_route_template_not_raw_path(client):
    """Path-parameter cardinality must not leak into metric labels."""
    client.get("/healthz")
    text = client.get("/metrics").text
    assert 'route="/healthz"' in text


def test_openapi_documents_the_api(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for path in (
        "/api/v1/extract",
        "/api/v1/kg/build",
        "/api/v1/kg/qa",
        "/api/v1/kg/qa/stream",
        "/api/v1/analysis/nl2sql",
    ):
        assert path in paths, f"{path} missing from OpenAPI schema"


def test_request_validation_rejects_empty_text(client):
    assert client.post("/api/v1/extract", json={"text": ""}).status_code == 422
    assert client.post("/api/v1/kg/qa", json={"question": "x", "hops": 9}).status_code == 422


def test_graph_name_cannot_escape_outputs(client):
    """Artefact names are caller-supplied, so path traversal must be refused."""
    response = client.post(
        "/api/v1/kg/qa", json={"question": "高血压", "graph_name": "../../.env"}
    )
    assert response.status_code in (400, 404)


def test_resolve_output_path_confines_to_outputs():
    from pathlib import Path

    from config.settings import OUTPUTS_DIR
    from service.deps import resolve_output_path

    assert resolve_output_path("g.json", ".json").parent == Path(OUTPUTS_DIR).resolve()
    # A directory component is stripped rather than honoured.
    assert resolve_output_path("../secret.json", ".json").parent == Path(OUTPUTS_DIR).resolve()
    # Suffix is pinned.
    assert resolve_output_path("g.txt", ".json").name.endswith(".json")
