# Contributing

1. Create a focused branch and keep commits small and reviewable.
2. Do not add raw patient data, secrets or unlicensed assets.
3. Add offline tests for behavior changes.
4. Run:

```powershell
python -m pytest -q
python -m compileall -q medigraph benchmarks mcp_server
python scripts/check_release.py
```

Metric changes must include the evaluation command, sample count, data split,
result JSON and checksum. Target values must never be presented as measured results.
