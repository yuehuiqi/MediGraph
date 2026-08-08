"""Client helper for the fine-tuned 0.8B OpenAI-compatible planner API."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


DEFAULT_API_URL = "http://127.0.0.1:18088/v1/chat/completions"
DEFAULT_MODEL = "qwen3p5-0p8b-orchestrator"


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("planner response is not a JSON object")
    return data


def plan_via_api(
    goal: str,
    api_url: str | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Turn a natural-language goal into an operator DAG through the LoRA API."""
    api_url = api_url or os.getenv("FINETUNED_ORCHESTRATOR_URL", DEFAULT_API_URL)
    model = model or os.getenv("FINETUNED_ORCHESTRATOR_MODEL", DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": goal}],
        "stream": False,
        "temperature": 0,
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    data = _parse_json_object(content)
    dag = data.get("dag", [])
    if not isinstance(dag, list) or not dag:
        raise ValueError("fine-tuned planner returned an empty DAG")
    return [node for node in dag if isinstance(node, dict)]

