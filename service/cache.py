"""Redis-backed result cache for idempotent, expensive endpoints.

What gets cached and why
------------------------
Extraction (~7 s: neural forward passes + optional LLM fallback) and NL2SQL
(LLM-generated path) are pure functions of their inputs *and of the models that
produced them*. That second part is the key-design point:

    medigraph:{namespace}:{sha256(inputs)}:{model_fingerprint}

The model fingerprint (extraction backend + LLM model + artefact versions) is part
of the key, not an afterthought: swapping `EXTRACTION_BACKEND` or upgrading the
LLM must miss the cache, otherwise the service silently serves results from a
model that is no longer deployed. TTL alone does not fix that class of staleness.

Failure policy
--------------
The cache is an optimisation, never a dependency: if Redis is down or the payload
fails to serialise, requests proceed uncached. After a connection error the client
backs off (`_RETRY_SECONDS`) instead of paying a connect timeout on every request.
Hits/misses/errors land in the `medigraph_cache_events_total` metric, so the hit
rate is observable rather than anecdotal.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any

from config.settings import get_cache_config, get_extraction_config, get_llm_config
from service.observability import CACHE_EVENTS

log = logging.getLogger("medigraph.cache")

_RETRY_SECONDS = 30.0


def model_fingerprint() -> str:
    """Version string of everything that determines a cached result's content."""
    llm = get_llm_config()
    extraction = get_extraction_config()
    raw = "|".join(
        (
            llm.provider,
            llm.model,
            extraction.backend,
            str(extraction.route_threshold),
            "v1",  # bump to invalidate the whole cache on a semantic change
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class ResultCache:
    """Namespaced JSON cache over Redis with graceful degradation."""

    def __init__(self, url: str | None = None, ttl_seconds: int | None = None, enabled: bool | None = None):
        config = get_cache_config()
        self.url = url or config.url
        self.ttl = ttl_seconds or config.ttl_seconds
        self.enabled = config.enabled if enabled is None else enabled
        self._client: Any = None
        self._lock = threading.Lock()
        self._down_until = 0.0

    # ------------------------------------------------------------------ #
    def _redis(self):
        if not self.enabled or time.monotonic() < self._down_until:
            return None
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import redis

                        self._client = redis.Redis.from_url(
                            self.url,
                            socket_timeout=1.0,
                            socket_connect_timeout=1.0,
                            decode_responses=True,
                        )
                        self._client.ping()
                        log.info("cache connected: %s", self.url)
                    except Exception as exc:  # noqa: BLE001 - cache is optional
                        log.warning("cache unavailable (%s); degrading to uncached", exc)
                        self._client = None
                        self._down_until = time.monotonic() + _RETRY_SECONDS
                        return None
        return self._client

    @staticmethod
    def key(namespace: str, payload: dict) -> str:
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"medigraph:{namespace}:{digest[:32]}:{model_fingerprint()}"

    # ------------------------------------------------------------------ #
    def get(self, namespace: str, payload: dict) -> dict | None:
        client = self._redis()
        if client is None:
            return None
        try:
            raw = client.get(self.key(namespace, payload))
        except Exception as exc:  # noqa: BLE001
            log.warning("cache get failed (%s)", exc)
            self._down_until = time.monotonic() + _RETRY_SECONDS
            CACHE_EVENTS.labels(namespace, "error").inc()
            return None
        if raw is None:
            CACHE_EVENTS.labels(namespace, "miss").inc()
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            CACHE_EVENTS.labels(namespace, "error").inc()
            return None
        CACHE_EVENTS.labels(namespace, "hit").inc()
        return value

    def set(self, namespace: str, payload: dict, value: dict) -> None:
        client = self._redis()
        if client is None:
            return
        try:
            client.set(
                self.key(namespace, payload),
                json.dumps(value, ensure_ascii=False),
                ex=self.ttl,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a request over the cache
            log.warning("cache set failed (%s)", exc)
            self._down_until = time.monotonic() + _RETRY_SECONDS

    def stats(self) -> dict:
        client = self._redis()
        if client is None:
            return {"enabled": self.enabled, "connected": False}
        try:
            info = client.info("stats")
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            return {
                "enabled": True,
                "connected": True,
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
            }
        except Exception:  # noqa: BLE001
            return {"enabled": self.enabled, "connected": False}


#: Process-wide instance; routers import this.
cache = ResultCache()
