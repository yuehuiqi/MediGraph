"""Lazily-built, process-wide singletons for expensive resources.

The offline entry points (demos, benchmarks) each construct their own extractor and
graph store because they run once and exit. A long-lived service cannot: loading
`graph_scaled.json` is ~88 MB of JSON and the neural extractor pulls a transformer
onto the device. Both are built on first use and reused afterwards, behind a lock
so a burst of concurrent requests does not trigger duplicate loads.

Everything here degrades instead of raising: a missing API key, absent model
weights or an unbuilt analytics DB must surface as a `/readyz` failure and a clean
503, not a stack trace at import time.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from config.settings import OUTPUTS_DIR, get_llm_config
from medigraph.graph.local_store import LocalGraphStore
from medigraph.operators.base import load_default_operators

log = logging.getLogger("medigraph.deps")

#: Reentrant on purpose: these builders compose (ensure_operators needs get_llm),
#: and a plain Lock self-deadlocks the moment one holder calls another.
_lock = threading.RLock()
_llm = None
_operators_loaded = False
_graphs: dict[str, LocalGraphStore] = {}
_nl2sql = None

ANALYTICS_DB = Path(OUTPUTS_DIR) / "analytics.db"


def get_llm():
    """Shared LLMClient. One instance keeps connection reuse and stats aggregation."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                from medigraph.llm.client import LLMClient

                _llm = LLMClient()
    return _llm


def ensure_operators() -> None:
    """Register the built-in operators once, sharing the service's LLM client."""
    global _operators_loaded
    if _operators_loaded:
        return
    # Build the client before taking the lock: it does its own locking, and this
    # keeps the critical section down to the registration itself.
    llm = get_llm()
    with _lock:
        if not _operators_loaded:
            load_default_operators(llm=llm)
            _operators_loaded = True


def resolve_output_path(name: str, suffix: str) -> Path:
    """Confine a caller-supplied artefact name to `outputs/`.

    The API takes graph names as plain strings, so this strips any directory part
    and pins the suffix -- otherwise `graph_name="../../.env"` would be a path
    traversal straight out of the artefact directory.
    """
    stem = Path(str(name)).name or f"api{suffix}"
    if not stem.endswith(suffix):
        stem = f"{Path(stem).stem}{suffix}"
    resolved = (Path(OUTPUTS_DIR) / stem).resolve()
    outputs_root = Path(OUTPUTS_DIR).resolve()
    if outputs_root not in resolved.parents and resolved != outputs_root:
        raise ValueError(f"refusing path outside outputs/: {name}")
    return resolved


def get_graph(graph_name: str = "graph.json") -> LocalGraphStore:
    """Load and cache a graph by artefact name."""
    path = resolve_output_path(graph_name, ".json")
    key = str(path)
    store = _graphs.get(key)
    if store is None:
        with _lock:
            store = _graphs.get(key)
            if store is None:
                if not path.exists():
                    raise FileNotFoundError(f"graph not found: {path.name}")
                log.info("loading graph %s", path.name)
                store = LocalGraphStore.load_json(path)
                _graphs[key] = store
                log.info(
                    "graph loaded: %s nodes=%d edges=%d",
                    path.name,
                    store.g.number_of_nodes(),
                    store.g.number_of_edges(),
                )
    return store


def get_nl2sql():
    """Shared NL2SQL engine (its schema linker loads the whole value vocabulary)."""
    global _nl2sql
    if _nl2sql is None:
        with _lock:
            if _nl2sql is None:
                from medigraph.analysis.nl2sql import NL2SQL

                if not ANALYTICS_DB.exists():
                    raise FileNotFoundError(
                        "analytics.db not built; run demos/demo_task3_complete.py first"
                    )
                _nl2sql = NL2SQL(str(ANALYTICS_DB), llm=get_llm())
    return _nl2sql


# --------------------------------------------------------------------------- #
# readiness
# --------------------------------------------------------------------------- #
def readiness_checks() -> list[dict]:
    """Probe each optional dependency without importing heavy modules eagerly."""
    checks: list[dict] = []

    config = get_llm_config()
    checks.append(
        {
            "name": "llm_api_key",
            "ready": bool(config.api_key),
            "detail": f"provider={config.provider} model={config.model}"
            if config.api_key
            else "no API key in .env; LLM-backed routes will fail",
        }
    )

    checks.append(
        {
            "name": "analytics_db",
            "ready": ANALYTICS_DB.exists(),
            "detail": str(ANALYTICS_DB) if ANALYTICS_DB.exists() else "not built",
        }
    )

    for graph_name in ("graph.json", "graph_scaled.json"):
        path = Path(OUTPUTS_DIR) / graph_name
        checks.append(
            {
                "name": f"graph:{graph_name}",
                "ready": path.exists(),
                "detail": f"{path.stat().st_size // 1024} KiB" if path.exists() else "absent",
            }
        )

    from config.settings import get_analytics_config, get_cache_config

    analytics = get_analytics_config()
    if analytics.backend == "postgres":
        try:
            from medigraph.analysis.pg_relational import ping

            ok = ping(analytics)
            checks.append(
                {
                    "name": "postgres",
                    "ready": ok,
                    "detail": f"{analytics.pg_host}:{analytics.pg_port}/{analytics.pg_database}"
                    if ok
                    else "unreachable; NL2SQL falls back to error responses",
                }
            )
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": "postgres", "ready": False, "detail": str(exc)[:200]})

    if get_cache_config().enabled:
        from service.cache import cache

        stats = cache.stats()
        checks.append(
            {
                "name": "redis_cache",
                # An unreachable cache degrades to uncached, so it is reported but
                # does not fail readiness.
                "ready": True,
                "detail": "connected" if stats.get("connected") else "unavailable (degraded to uncached)",
            }
        )

    try:
        from medigraph.extraction.cascade import load_neural_extractor

        extractor = load_neural_extractor()
        checks.append(
            {
                "name": "neural_extractor",
                "ready": extractor is not None,
                "detail": "loaded" if extractor is not None else "unavailable; cascade falls back to lexicon/LLM",
            }
        )
    except Exception as exc:  # noqa: BLE001 - optional dependency
        checks.append({"name": "neural_extractor", "ready": False, "detail": str(exc)[:200]})

    return checks
