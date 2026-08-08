"""DataMate REST client.

Wraps the DataMate APIs needed to (1) upload custom operators, (2) create a
cleaning template (operator DAG), (3) create + run a cleaning task on a dataset,
and (4) manage datasets/files. Everything routes through the DataMate gateway
(default http://localhost:8080/api); JWT is disabled by default in the dev deploy.

Endpoints/fields verified against the live DataMate OpenAPI:
  operators:  /api/operators/upload/{pre-upload,chunk}, /api/operators/upload
  templates:  /api/cleaning/templates
  tasks:      /api/cleaning/tasks  (POST auto-executes), /api/cleaning/tasks/{id}
  datasets:   /api/data-management/datasets (+ /files/upload/{pre-upload,chunk})
Response envelope: {"code":"0","message":..., "data":...}. Bodies are camelCase.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests


class DataMateError(RuntimeError):
    pass


class DataMateClient:
    def __init__(self, base_url: str | None = None, timeout: int = 60):
        self.base = (base_url or os.getenv("DATAMATE_BASE", "http://localhost:8080/api")).rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    # ------------------------------------------------------------------ #
    def _unwrap(self, resp: requests.Response) -> Any:
        if resp.status_code != 200:
            raise DataMateError(f"{resp.request.method} {resp.url} -> HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            body = resp.json()
        except ValueError:
            return resp.text
        if isinstance(body, dict) and "code" in body:
            if str(body.get("code")) not in ("0", "200"):
                raise DataMateError(f"{resp.url} -> code={body.get('code')} msg={body.get('message')}")
            return body.get("data")
        return body

    def _get(self, path: str, **kw):
        return self._unwrap(self.s.get(f"{self.base}{path}", timeout=self.timeout, **kw))

    def _post(self, path: str, json: Any = None, **kw):
        return self._unwrap(self.s.post(f"{self.base}{path}", json=json, timeout=self.timeout, **kw))

    def _put(self, path: str, json: Any = None, **kw):
        return self._unwrap(self.s.put(f"{self.base}{path}", json=json, timeout=self.timeout, **kw))

    # ---- operators --------------------------------------------------- #
    def upload_operator(self, package_path: str | Path) -> dict:
        """Upload and create/update an operator package. Returns OperatorDto."""
        package_path = Path(package_path)
        data = package_path.read_bytes()
        file_name = package_path.name

        req_id = self._post("/operators/upload/pre-upload", json={})
        if isinstance(req_id, dict):
            req_id = req_id.get("reqId") or req_id.get("id") or req_id.get("req_id")
        if not req_id:
            raise DataMateError("pre-upload returned no reqId")

        files = {"file": (file_name, data, "application/octet-stream")}
        form = {"reqId": str(req_id), "fileNo": "1", "fileName": file_name,
                "totalChunkNum": "1", "chunkNo": "1"}
        self._unwrap(self.s.post(f"{self.base}/operators/upload/chunk",
                                 data=form, files=files, timeout=self.timeout))
        parsed = self._post("/operators/upload", json={"fileName": file_name})
        if not isinstance(parsed, dict) or not parsed.get("id"):
            raise DataMateError("operator archive parser returned no operator id")

        operator_id = parsed["id"]
        detail = self.s.get(f"{self.base}/operators/{operator_id}", timeout=self.timeout)
        if detail.status_code == 404:
            # /operators/upload only parses the archive; a new operator must be
            # persisted explicitly through /operators/create.
            return self._post("/operators/create", json=parsed)
        self._unwrap(detail)

        # Updating with fileName makes DataMate extract the newly uploaded
        # archive into the shared runtime volume.
        allowed = {
            "name", "description", "version", "inputs", "outputs", "runtime",
            "settings", "fileName", "fileSize", "metrics", "usageCount",
            "isStar", "categories", "overrides", "requirements", "readme",
            "releases",
        }
        update = {key: value for key, value in parsed.items() if key in allowed and value is not None}
        return self.update_operator(operator_id, **update)

    def list_operators(self, keyword: str = "", page: int = 0, size: int = 50) -> list[dict]:
        data = self._post("/operators/list", json={"page": page, "size": size, "keyword": keyword})
        return (data or {}).get("content", []) if isinstance(data, dict) else (data or [])

    def update_operator(self, operator_id: str, **fields: Any) -> dict:
        return self._put(f"/operators/{operator_id}", json=fields)

    # ---- datasets ---------------------------------------------------- #
    def create_dataset(self, name: str, ds_type: str = "TEXT", description: str = "") -> dict:
        return self._post("/data-management/datasets",
                          json={"name": name, "datasetType": ds_type, "description": description})

    def upload_file_to_dataset(self, dataset_id: str, file_path: str | Path, prefix: str = "") -> None:
        """Single-file chunked upload into a dataset."""
        file_path = Path(file_path)
        data = file_path.read_bytes()
        pre = self._post(
            f"/data-management/datasets/{dataset_id}/files/upload/pre-upload",
            json={"hasArchive": False, "totalFileNum": 1, "totalSize": len(data), "prefix": prefix},
        )
        req_id = pre.get("reqId") if isinstance(pre, dict) else pre
        files = {"file": (file_path.name, data, "application/octet-stream")}
        form = {"reqId": str(req_id), "fileNo": "1", "fileName": file_path.name,
                "totalChunkNum": "1", "chunkNo": "1", "prefix": prefix}
        self._unwrap(self.s.post(
            f"{self.base}/data-management/datasets/{dataset_id}/files/upload/chunk",
            data=form, files=files, timeout=self.timeout))

    def count_dataset_files(self, dataset_id: str) -> int:
        data = self._get(f"/data-management/datasets/{dataset_id}/files?page=0&size=1")
        return (data or {}).get("totalElements", 0) if isinstance(data, dict) else 0

    def wait_dataset_files(self, dataset_id: str, expected: int, interval: float = 2.0, max_wait: float = 60.0) -> int:
        """Poll until the dataset has at least `expected` files committed (avoids a task/upload race)."""
        waited = 0.0
        while waited < max_wait:
            n = self.count_dataset_files(dataset_id)
            if n >= expected:
                return n
            time.sleep(interval)
            waited += interval
        return self.count_dataset_files(dataset_id)

    # ---- cleaning templates / tasks ---------------------------------- #
    def create_template(self, name: str, description: str, instance: list[dict]) -> dict:
        return self._post("/cleaning/templates",
                          json={"name": name, "description": description, "instance": instance})

    def create_task(self, name: str, src_dataset_id: str, src_dataset_name: str,
                    dest_dataset_name: str, dest_dataset_type: str = "TEXT",
                    template_id: str | None = None, instance: list[dict] | None = None,
                    description: str = "") -> dict:
        """Create a cleaning task. DataMate auto-executes on create."""
        body: dict[str, Any] = {
            "name": name, "description": description,
            "srcDatasetId": src_dataset_id, "srcDatasetName": src_dataset_name,
            "destDatasetName": dest_dataset_name, "destDatasetType": dest_dataset_type,
        }
        if template_id:
            body["templateId"] = template_id
        if instance:
            body["instance"] = instance
        return self._post("/cleaning/tasks", json=body)

    def get_task(self, task_id: str) -> dict:
        return self._get(f"/cleaning/tasks/{task_id}")

    def poll_task(self, task_id: str, interval: float = 3.0, max_wait: float = 600.0) -> dict:
        """Poll until task reaches a terminal status."""
        terminal = {"COMPLETED", "PARTIAL_SUCCESS", "STOPPED", "FAILED"}
        waited = 0.0
        while waited < max_wait:
            task = self.get_task(task_id)
            status = (task or {}).get("status", "")
            prog = (task or {}).get("progress", {}) or {}
            print(f"  [task {task_id[:8]}] status={status} progress={prog.get('process', 0)}%")
            if status in terminal:
                return task
            time.sleep(interval)
            waited += interval
        return self.get_task(task_id)

    def task_log(self, task_id: str, retry_count: int = 0) -> list[dict]:
        try:
            return self._get(f"/cleaning/tasks/{task_id}/log/{retry_count}") or []
        except Exception:  # noqa: BLE001
            return []

    def task_result(self, task_id: str) -> list[dict]:
        return self._get(f"/cleaning/tasks/{task_id}/result") or []

    def download_result(self, task_id: str, out_path: str | Path) -> str:
        resp = self.s.get(f"{self.base}/cleaning/tasks/{task_id}/result/download", timeout=self.timeout)
        if resp.status_code != 200:
            raise DataMateError(f"download result HTTP {resp.status_code}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return str(out_path)
