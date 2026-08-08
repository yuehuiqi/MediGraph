"""OpenAI-compatible API server for the fine-tuned 0.8B DAG orchestrator.

This intentionally avoids a vLLM dependency so it can run on Windows with the
same `meditune` conda environment used for LoRA training/evaluation. It exposes
the two endpoints Nexent needs:

  GET  /v1/models
  POST /v1/chat/completions

The model loaded here is:
  base:    Qwen/Qwen3.5-0.8B
  adapter: finetune/outputs/qwen3p5-0p8b-orchestrator

Example:
  python finetune/openai_compatible_server.py --host 0.0.0.0 --port 18088
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune.infer import DEFAULT_ADAPTER, LocalOrchestrator  # noqa: E402


MODEL_ID = "qwen3p5-0p8b-orchestrator"
SERVER_STATE: dict[str, Any] = {
    "orchestrator": None,
    "base_path": None,
    "adapter_path": None,
    "started_at": int(time.time()),
}


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _extract_goal(messages: list[dict[str, Any]] | None, prompt: str | None = None) -> str:
    if prompt:
        return prompt
    messages = messages or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                return "\n".join(parts)
    return ""


def _completion_payload(content: str, created: int, request_id: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _chunk_payload(content: str, created: int, request_id: str, final: bool = False) -> dict[str, Any]:
    if final:
        choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [choice],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "MediGraphLoRAOpenAI/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()

    def _send_json(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": MODEL_ID,
                    "base_path": SERVER_STATE["base_path"],
                    "adapter_path": SERVER_STATE["adapter_path"],
                    "loaded": SERVER_STATE["orchestrator"] is not None,
                },
            )
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": SERVER_STATE["started_at"],
                            "owned_by": "MediGraph",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": f"Not found: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/v1/completions"}:
            self._send_json(404, {"error": {"message": f"Not found: {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            goal = _extract_goal(payload.get("messages"), payload.get("prompt"))
            if not goal:
                raise ValueError("No user message or prompt found")

            orchestrator: LocalOrchestrator = SERVER_STATE["orchestrator"]
            dag = orchestrator.plan(goal)
            content = json.dumps({"dag": dag}, ensure_ascii=False, indent=2)
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                for chunk in (content, ""):
                    packet = _chunk_payload(chunk, created, request_id, final=(chunk == ""))
                    self.wfile.write(b"data: " + _json_bytes(packet) + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._send_json(200, _completion_payload(content, created, request_id))
        except Exception as exc:  # noqa: BLE001 - API surface should return JSON errors
            self._send_json(500, {"error": {"message": str(exc), "type": exc.__class__.__name__}})


def resolve_base_path(base: str | None) -> str:
    if base:
        return base
    from modelscope import snapshot_download

    return snapshot_download("Qwen/Qwen3.5-0.8B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18088)
    parser.add_argument("--base", default=None, help="Local base model path. Defaults to ModelScope cache.")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    base_path = resolve_base_path(args.base)
    adapter_path = str(Path(args.adapter).resolve()) if args.adapter else None
    print(f"[startup] loading {MODEL_ID}")
    print(f"[startup] base={base_path}")
    print(f"[startup] adapter={adapter_path}")
    sys.stdout.flush()

    SERVER_STATE["orchestrator"] = LocalOrchestrator(
        base_path=base_path,
        adapter_path=adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    SERVER_STATE["base_path"] = base_path
    SERVER_STATE["adapter_path"] = adapter_path

    print(f"[startup] loaded. serving http://{args.host}:{args.port}/v1")
    sys.stdout.flush()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
