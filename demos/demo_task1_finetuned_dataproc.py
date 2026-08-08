"""Task 1 end-to-end demo using the fine-tuned 0.8B model as DAG planner.

Start the local model API first:
  powershell -ExecutionPolicy Bypass -File scripts/start_finetuned_model.ps1

Then run:
  python demos/demo_task1_finetuned_dataproc.py --max-docs 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, RAW_DEMO_DIR  # noqa: E402
from finetune.api_planner import DEFAULT_MODEL, plan_via_api  # noqa: E402
from medigraph.agents.dataproc_agent import DataProcAgent  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402
from medigraph.utils.io import iter_documents, write_json, write_jsonl  # noqa: E402

enable_utf8()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task1 demo with fine-tuned 0.8B planner")
    parser.add_argument("--input", default=str(RAW_DEMO_DIR))
    parser.add_argument(
        "--goal",
        default="清洗医疗文档、切块、抽取实体和关系，最后校验三元组并输出结构化结果",
    )
    parser.add_argument("--max-docs", type=int, default=1)
    args = parser.parse_args()

    docs = iter_documents(args.input)[: args.max_docs]
    if not docs:
        raise SystemExit(f"No documents found under {args.input}")

    print(f"Planner model: {DEFAULT_MODEL}")
    agent = DataProcAgent(local_planner=plan_via_api)
    records = []
    last_result = None
    for doc in docs:
        print(f"\n=== Processing: {doc['fileName']} ===")
        last_result = agent.run(args.goal, {"text": doc["text"]})
        payload = last_result["payload"]
        records.append(
            {
                "fileName": doc["fileName"],
                "entities": payload.get("entities", []),
                "valid_triples": payload.get("valid", []),
            }
        )

    processed_path = write_jsonl(records, OUTPUTS_DIR / "task1_finetuned_processed.jsonl")
    report_path = write_json(
        {
            "planner_model": DEFAULT_MODEL,
            "goal": args.goal,
            "dag": last_result["dag"],
            "report": last_result["report"],
            "lineage": last_result["lineage"],
        },
        OUTPUTS_DIR / "task1_finetuned_report.json",
    )
    print("\n========== FINETUNED PLANNER RESULT ==========")
    print(f"Processed records -> {processed_path}")
    print(f"Report + DAG      -> {report_path}")


if __name__ == "__main__":
    main()
