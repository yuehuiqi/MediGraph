"""Create a real Nexent web conversation for the synthetic acceptance case.

This script uses the same HTTP endpoints as the browser UI. The conversation,
user messages, tool calls, and assistant replies are therefore persisted and
immediately visible on the Nexent chat page.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "data" / "demo_cases" / "cardiometabolic_20260701"
OUTPUT_DIR = ROOT / "outputs"
BASE_URL = os.getenv("NEXENT_WEB_URL", "http://localhost:3000").rstrip("/")
USER_ID = os.getenv("NEXENT_USER_ID", "3233c4d4-fa1e-4685-ab76-ba3529e9df17")
AGENT_ID = int(os.getenv("NEXENT_AGENT_ID", "5"))
MODEL_ID = int(os.getenv("NEXENT_MODEL_ID", "5"))
VERSION_NO = int(os.getenv("NEXENT_AGENT_VERSION", "1"))


def _session_token() -> str:
    code = (
        "from utils.auth_utils import generate_session_jwt;"
        f"print(generate_session_jwt({USER_ID!r}, expires_in=7200))"
    )
    proc = subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            "/opt/backend",
            "nexent-config",
            "python",
            "-c",
            code,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    token = proc.stdout.strip()
    if not token:
        raise RuntimeError("Nexent session token generation returned an empty value")
    return token


def _request_json(
    session: requests.Session,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> Any:
    response = session.request(
        method,
        f"{BASE_URL}{path}",
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("code") not in (None, 0):
        raise RuntimeError(f"Nexent API error at {path}: {payload}")
    return payload


def _run_turn(
    session: requests.Session,
    conversation_id: int,
    query: str,
    turn_no: int,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    payload = {
        "query": query,
        "conversation_id": conversation_id,
        "history": history,
        "minio_files": None,
        "agent_id": AGENT_ID,
        "model_id": MODEL_ID,
        "version_no": VERSION_NO,
        "is_debug": False,
    }
    started = time.time()
    response = session.post(
        f"{BASE_URL}/api/agent/run",
        json=payload,
        stream=True,
        timeout=(60, 900),
    )
    response.raise_for_status()
    raw_path = OUTPUT_DIR / f"nexent_new_case_stream_{turn_no}.txt"
    chunks: list[str] = []
    with raw_path.open("w", encoding="utf-8") as stream_file:
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
            chunks.append(line)
            stream_file.write(line + "\n")
            stream_file.flush()
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    event = {}
                if event.get("type") == "final_answer":
                    break
    response.close()
    return {
        "turn": turn_no,
        "query": query,
        "http_status": response.status_code,
        "elapsed_seconds": round(time.time() - started, 2),
        "stream_file": str(raw_path),
        "stream_lines": len(chunks),
        "stream_tail": chunks[-20:],
    }


def main() -> None:
    documents = []
    for path in sorted(CASE_DIR.glob("*.txt")):
        documents.append(f"【{path.name}】\n{path.read_text(encoding='utf-8').strip()}")
    if not documents:
        raise SystemExit(f"No case documents found under {CASE_DIR}")
    case_text = "\n\n".join(documents)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {_session_token()}",
            "Content-Type": "application/json",
            "User-Agent": "MediGraph-Acceptance/1.0",
            "Accept-Language": "zh-CN",
        }
    )
    existing_id = os.getenv("NEXENT_EXISTING_CONVERSATION_ID", "").strip()
    if existing_id:
        conversation_id = int(existing_id)
        detail = _request_json(session, "GET", f"/api/conversation/{conversation_id}")
        turns = []
        for turn_no in range(1, 4):
            raw_path = OUTPUT_DIR / f"nexent_new_case_stream_{turn_no}.txt"
            turns.append(
                {
                    "turn": turn_no,
                    "stream_file": str(raw_path),
                    "stream_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
                }
            )
        summary = {
            "ok": True,
            "base_url": BASE_URL,
            "web_url": f"{BASE_URL}/zh/chat?conversationId={conversation_id}",
            "conversation": {"conversation_id": conversation_id},
            "agent": {
                "agent_id": AGENT_ID,
                "model_id": MODEL_ID,
                "version_no": VERSION_NO,
            },
            "turns": turns,
            "detail": detail,
        }
        out = OUTPUT_DIR / "nexent_new_case_summary.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "conversation_id": conversation_id, "summary": str(out)}, ensure_ascii=False))
        return

    created = _request_json(
        session,
        "PUT",
        "/api/conversation/create",
        body={"title": "2026-07-01 心血管代谢新案例验收"},
    )
    conversation = created["data"]
    conversation_id = int(conversation["conversation_id"])
    print(f"conversation_id={conversation_id}", flush=True)

    turn1 = (
        "请完成一次可录屏的 MediGraph 新案例处理。先调用 inspect_extraction_models 审计当前"
        "神经 GPLinker 抽取级联，再调用 build_medical_kg 将下列三份合成、非患者数据构建为"
        "独立图谱 new_case_cardiometabolic_20260701.json。请简洁汇报模型路由、实体数、"
        "候选/有效三元组数、校验通过率和图谱文件。\n\n"
        f"{case_text}"
    )
    turn2 = (
        "请对图谱 new_case_cardiometabolic_20260701.json 连续调用 medical_kg_qa 回答两个问题："
        "（1）心力衰竭有哪些典型症状和检查依据？"
        "（2）高血压可能并发哪些疾病？"
        "每个答案都列出证据三元组、置信度和来源；不得补充图谱中不存在的医学结论。"
    )
    turn3 = (
        "请验收刚更新的图谱驱动分析能力：先调用 inspect_analysis_assets 审计图谱复用、"
        "NL2SQL 双库执行准确率和数据表规模；再调用 analyze_medical_data 回答"
        "“哪个科室接诊的不同病人数量最多？”。请给出实际路由、SQL、查询结果、"
        "数据血缘和生成的 HTML 报告路径。"
    )
    first_turn = _run_turn(session, conversation_id, turn1, 1, [])
    prior_history = [
        {"role": "user", "content": turn1},
        {
            "role": "assistant",
            "content": (
                "已完成抽取级联审计与独立图谱构建；"
                "后续问题请查询 new_case_cardiometabolic_20260701.json。"
            ),
        },
    ]
    second_turn = _run_turn(
        session,
        conversation_id,
        turn2,
        2,
        prior_history,
    )
    analysis_history = [
        *prior_history,
        {"role": "user", "content": turn2},
        {
            "role": "assistant",
            "content": (
                "已基于独立图谱完成两项可溯源问答；"
                "现继续验收更新后的图谱驱动分析能力。"
            ),
        },
    ]
    third_turn = _run_turn(
        session,
        conversation_id,
        turn3,
        3,
        analysis_history,
    )
    turns = [first_turn, second_turn, third_turn]
    detail = _request_json(session, "GET", f"/api/conversation/{conversation_id}")
    summary = {
        "ok": True,
        "base_url": BASE_URL,
        "web_url": f"{BASE_URL}/zh/chat?conversationId={conversation_id}",
        "conversation": conversation,
        "agent": {
            "agent_id": AGENT_ID,
            "model_id": MODEL_ID,
            "version_no": VERSION_NO,
        },
        "turns": turns,
        "detail": detail,
    }
    out = OUTPUT_DIR / "nexent_new_case_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "conversation_id": conversation_id, "summary": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Nexent new-case run failed: {exc}", file=sys.stderr, flush=True)
        raise
