"""API demo for the fine-tuned <1B DAG orchestrator.

Prerequisite:
  python finetune/openai_compatible_server.py --host 0.0.0.0 --port 18088

Then run:
  python demos/demo_finetuned_orchestrator_api.py
"""
from __future__ import annotations

import json
import urllib.request


API_URL = "http://localhost:18088/v1/chat/completions"


def main() -> None:
    payload = {
        "model": "qwen3p5-0p8b-orchestrator",
        "messages": [
            {
                "role": "user",
                "content": "清洗病理文档、切块、抽取实体和关系，最后做三元组校验",
            }
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    print("=== Fine-tuned 0.8B orchestrator output ===")
    print(content)


if __name__ == "__main__":
    main()
