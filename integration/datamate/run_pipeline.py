"""Run a medical data-processing pipeline INSIDE DataMate via REST.

End-to-end (this is "Task 1 actually running as a DataMate operator pipeline"):
  1. create a dataset
  2. upload medical documents (from CCF/data/corpus) into it
  3. create a cleaning template = operator DAG
     (text_clean -> chunker -> medical_ner -> medical_re -> triple_validator)
  4. create a cleaning task (DataMate auto-executes it) and poll to completion
  5. download the processed results

Requires upload_operators.py to have run first (uses outputs/datamate_operator_ids.json).

Usage:
  python integration/datamate/run_pipeline.py --input data/corpus --max-files 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import OUTPUTS_DIR, RAW_DEMO_DIR  # noqa: E402
from integration.datamate.datamate_client import DataMateClient  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

enable_utf8()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--input", default=str(RAW_DEMO_DIR))
    ap.add_argument("--max-files", type=int, default=3)
    ap.add_argument("--ops", default="text_clean,chunker,medical_ner,medical_re,triple_validator",
                    help="operator DAG order (comma-separated)")
    args = ap.parse_args()

    ids_file = OUTPUTS_DIR / "datamate_operator_ids.json"
    if not ids_file.exists():
        print("operator IDs not found. Run upload_operators.py first.")
        sys.exit(1)
    op_ids = json.loads(ids_file.read_text(encoding="utf-8"))

    client = DataMateClient(base_url=args.base)
    stamp = time.strftime("%m%d_%H%M%S")

    # 1) dataset
    ds = client.create_dataset(f"medai_src_{stamp}", "TEXT", "MediGraph pipeline input")
    src_id = ds.get("id")
    src_name = ds.get("name")
    print(f"[1] dataset created: {src_id} ({src_name})")

    # 2) upload docs
    docs = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in (".md", ".txt"))[: args.max_files]
    if not docs:
        print(f"no .md/.txt files under {args.input}")
        sys.exit(1)
    for d in docs:
        client.upload_file_to_dataset(src_id, d)
        print(f"    uploaded {d.name}")
    # wait until files are committed before starting the task (avoids race -> 0 files scanned)
    n = client.wait_dataset_files(src_id, expected=len(docs))
    print(f"    dataset now has {n} committed file(s)")

    # API creds for the LLM-backed operators (NER/RE) to run inside DataMate
    from config.settings import get_llm_config
    cfg = get_llm_config()
    llm_overrides = {"apiBase": cfg.base_url, "apiKey": cfg.api_key, "model": cfg.model}

    # 3) template = operator DAG
    instance = []
    for name in [o.strip() for o in args.ops.split(",") if o.strip()]:
        oid = op_ids.get(name)
        if not oid:
            print(f"    ! no operator id for {name}; skipped")
            continue
        overrides = dict(llm_overrides) if name in ("medical_ner", "medical_re") else {}
        # inputs/outputs types are required by the template validator; our ops are text->text
        instance.append({"id": oid, "name": name, "inputs": "text", "outputs": "text", "overrides": overrides})
    tpl = client.create_template(
        f"medai_tpl_{stamp}",
        "Clean->Chunk->NER->RE->Validator",
        instance,
    )
    tpl_id = tpl.get("id")
    print(f"[3] template created: {tpl_id} with {len(instance)} operators")

    # 4) task (auto-executes) + poll. Submit the expanded instances directly:
    # the current backend accepts templateId but can create an empty process.
    task = client.create_task(
        name=f"medai_task_{stamp}", src_dataset_id=src_id, src_dataset_name=src_name,
        dest_dataset_name=f"medai_out_{stamp}", dest_dataset_type="TEXT", instance=instance,
        description="MediGraph medical extraction pipeline",
    )
    task_id = task.get("id")
    print(f"[4] task created: {task_id} (auto-executing) ...")
    final = client.poll_task(task_id)
    print(f"    final status: {final.get('status')}")
    if final.get("status") != "COMPLETED":
        print("    --- last task log lines (diagnostics) ---")
        for entry in client.task_log(task_id)[-8:]:
            print(f"      {entry.get('level','')}: {str(entry.get('message',''))[:160]}")
        print("    Check the task traceback above and verify the five operator settings.")

    # 5) download results
    out_zip = OUTPUTS_DIR / "datamate_pipeline_result" / f"{task_id}.zip"
    try:
        client.download_result(task_id, out_zip)
        print(f"[5] results downloaded -> {out_zip}")
    except Exception as exc:  # noqa: BLE001
        print(f"[5] result download skipped: {exc}")

    summary = {"dataset_id": src_id, "template_id": tpl_id, "task_id": task_id,
               "status": final.get("status"), "progress": final.get("progress")}
    (OUTPUTS_DIR / "datamate_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary -> {OUTPUTS_DIR / 'datamate_pipeline_summary.json'}")


if __name__ == "__main__":
    main()
