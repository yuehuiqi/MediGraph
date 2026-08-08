"""Upload the 5 medical data-processing operators to DataMate via REST.

Packages CCF/datamate_ops/{text_clean,chunker,medical_ner,medical_re,
triple_validator} into zips
(reusing datamate_ops/build_zip.py) and chunked-uploads them to DataMate. Saves
the returned operator IDs to outputs/datamate_operator_ids.json for run_pipeline.

Usage (DataMate must be running):
  python integration/datamate/upload_operators.py
  python integration/datamate/upload_operators.py --base http://localhost:8080/api
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import OUTPUTS_DIR  # noqa: E402
from integration.datamate.datamate_client import DataMateClient  # noqa: E402
from medigraph.utils.console import enable_utf8  # noqa: E402

enable_utf8()
ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "datamate_ops" / "dist"
OPERATORS = ["text_clean", "chunker", "medical_ner", "medical_re", "triple_validator"]


def refresh_local_runtime(client: DataMateClient, name: str) -> bool:
    """Refresh the extracted operator volume after an in-place re-upload.

    DataMate currently replaces the uploaded zip for an existing raw_id but
    leaves the old extracted runtime files in place. Apply the uploaded archive
    to the shared volume when this script targets the local Docker deployment.
    """
    if urlparse(client.base).hostname not in {"localhost", "127.0.0.1", "::1"}:
        return False
    if not shutil.which("docker"):
        return False
    inspect = subprocess.run(
        ["docker", "inspect", "datamate-backend-python"],
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        return False
    subprocess.run(
        [
            "docker", "exec", "datamate-backend-python",
            "python", "-m", "zipfile", "-e",
            f"/operators/upload/{name}.zip",
            f"/operators/extract/{name}",
        ],
        check=True,
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="DataMate base URL, e.g. http://localhost:8080/api")
    args = ap.parse_args()

    # (re)build the operator zips
    print("Packaging operators ...")
    subprocess.run([sys.executable, str(ROOT / "datamate_ops" / "build_zip.py")], check=True)

    client = DataMateClient(base_url=args.base)
    ids: dict[str, str] = {}
    for name in OPERATORS:
        zip_path = DIST / f"{name}.zip"
        if not zip_path.exists():
            print(f"  ! missing {zip_path}, skipped")
            continue
        print(f"Uploading {name} ...")
        try:
            op = client.upload_operator(zip_path)
            op_id = op.get("id") if isinstance(op, dict) else None
            ids[name] = op_id
            print(f"  OK -> operator_id={op_id}  name={op.get('name') if isinstance(op, dict) else ''}")
            version = str(op.get("version", "")).strip() if isinstance(op, dict) else ""
            if refresh_local_runtime(client, name):
                print(f"  runtime refreshed -> {name} {version}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {name}: {exc}")

    out = OUTPUTS_DIR / "datamate_operator_ids.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved operator IDs -> {out}")
    print("Next: python integration/datamate/run_pipeline.py")


if __name__ == "__main__":
    main()
