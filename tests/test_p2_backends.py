"""Offline-safe tests for the P2 storage layer: PG relational, pgvector, Redis cache.

Everything that needs a live server is gated behind a reachability probe and skips
cleanly on CI / fresh clones. What can be tested purely (row-generation
determinism, cache-key semantics, graceful degradation) always runs.
"""
from __future__ import annotations

import socket

import pytest

from config.settings import get_analytics_config


def _port_open(host: str, port: int) -> bool:
    probe = socket.socket()
    probe.settimeout(1.5)
    try:
        probe.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


_config = get_analytics_config()
HAS_PSYCOPG = True
try:  # psycopg is in requirements but keep CI's minimal install green
    import psycopg  # noqa: F401
except ImportError:
    HAS_PSYCOPG = False

PG_UP = HAS_PSYCOPG and bool(_config.pg_password) and _port_open(_config.pg_host, _config.pg_port)
REDIS_UP = _port_open("127.0.0.1", 6380)

needs_pg = pytest.mark.skipif(not PG_UP, reason="PostgreSQL not reachable (medigraph-pg on 5433)")
needs_redis = pytest.mark.skipif(not REDIS_UP, reason="Redis not reachable (medigraph-redis on 6380)")


# --------------------------------------------------------------------------- #
# row generation: pure, runs everywhere
# --------------------------------------------------------------------------- #
def test_generate_rows_is_deterministic():
    """Both engines ingest these rows; determinism is what makes SQLite-vs-PG a
    logical-equivalence check instead of a comparison of two random datasets."""
    from medigraph.analysis.relational import generate_rows
    from medigraph.graph.local_store import LocalGraphStore

    store = LocalGraphStore()
    store.upsert_triple("高血压", "Disease", "treated_in_department", "心内科", "Department")
    store.upsert_triple("高血压", "Disease", "recommend_drug", "氨氯地平", "Drug")

    first = generate_rows(store, n_visits=50, seed=42)
    second = generate_rows(store, n_visits=50, seed=42)
    assert first == second
    assert len(first["patient_visits"]) == 50
    assert first != generate_rows(store, n_visits=50, seed=43)


# --------------------------------------------------------------------------- #
# PostgreSQL backend
# --------------------------------------------------------------------------- #
@needs_pg
def test_pg_readonly_rejects_writes_server_side():
    """The transaction itself must be read-only -- an FK error would mean the
    write was attempted and merely happened to fail."""
    from medigraph.analysis.pg_relational import execute_readonly_pg

    for sql in ("DELETE FROM kg_entities", "UPDATE patient_visits SET cost = 0",
                "INSERT INTO kg_entities VALUES ('x', 'y')"):
        _, _, error = execute_readonly_pg(sql)
        assert "read-only transaction" in error, (sql, error)


@needs_pg
def test_pg_statement_timeout_enforced():
    from medigraph.analysis.pg_relational import execute_readonly_pg

    _, _, error = execute_readonly_pg("SELECT pg_sleep(10)", timeout_seconds=1.0)
    assert "statement timeout" in error


@needs_pg
def test_pg_row_cap_enforced():
    from medigraph.analysis.pg_relational import execute_readonly_pg

    _, _, error = execute_readonly_pg("SELECT * FROM kg_triples", max_rows=10)
    assert "safety limit" in error


@needs_pg
def test_pg_readonly_does_not_leak_into_pool():
    """A build after a read-only call must still be able to write (regression for
    the connection-property leak the pooled reuse makes possible)."""
    from medigraph.analysis.pg_relational import execute_readonly_pg, get_pool

    _, _, error = execute_readonly_pg("SELECT 1")
    assert not error
    with get_pool().connection() as conn:
        conn.execute("CREATE TEMP TABLE _writable_probe (x INT)")
        conn.execute("INSERT INTO _writable_probe VALUES (1)")
        assert conn.execute("SELECT x FROM _writable_probe").fetchone() == (1,)
        conn.rollback()


@needs_pg
def test_sqlite_and_pg_agree_on_gold_queries():
    """Same generated rows + portable SQL -> identical multisets across engines."""
    import sqlite3

    from medigraph.analysis.pg_relational import execute_readonly_pg
    from medigraph.analysis.sql_guard import transpile
    from config.settings import OUTPUTS_DIR

    queries = [
        "SELECT COUNT(*) FROM patient_visits",
        "SELECT department, COUNT(*) AS cnt FROM patient_visits GROUP BY department ORDER BY cnt DESC, department",
        "SELECT AVG(age) FROM patient_visits WHERE disease = '高血压'",
        "SELECT drug, COUNT(*) AS cnt FROM prescriptions GROUP BY drug ORDER BY cnt DESC, drug LIMIT 5",
        "SELECT substr(visit_date,1,7) AS month, COUNT(*) FROM patient_visits GROUP BY month ORDER BY month",
    ]
    lite = sqlite3.connect(f"file:{OUTPUTS_DIR}/analytics.db?mode=ro", uri=True)
    for sql in queries:
        expected = [tuple(row) for row in lite.execute(sql).fetchall()]
        columns, rows, error = execute_readonly_pg(transpile(sql, write="postgres"))
        assert not error, (sql, error)
        actual = [
            tuple(float(v) if hasattr(v, "quantize") else v for v in row) for row in rows
        ]
        normalised_expected = [
            tuple(round(v, 6) if isinstance(v, float) else v for v in row) for row in expected
        ]
        normalised_actual = [
            tuple(round(v, 6) if isinstance(v, float) else v for v in row) for row in actual
        ]
        assert normalised_actual == normalised_expected, sql
    lite.close()


@needs_pg
def test_nl2sql_postgres_backend_matches_sqlite():
    from medigraph.analysis.nl2sql import NL2SQL
    from config.settings import OUTPUTS_DIR

    question = "各科室的就诊量是多少"
    lite = NL2SQL(f"{OUTPUTS_DIR}/analytics.db", backend="sqlite").query(question)
    pg = NL2SQL(f"{OUTPUTS_DIR}/analytics.db", backend="postgres").query(question)
    assert not lite["error"] and not pg["error"]
    assert lite["generation_mode"] == pg["generation_mode"] == "deterministic_template"
    # The template's ORDER BY has no tiebreaker, so tie order is engine-specific;
    # compare as multisets -- the logical result, not the incidental ordering.
    assert sorted(tuple(r) for r in lite["rows"]) == sorted(tuple(r) for r in pg["rows"])


# --------------------------------------------------------------------------- #
# pgvector
# --------------------------------------------------------------------------- #
@needs_pg
def test_pgvector_roundtrip_and_self_recall():
    import numpy as np

    from medigraph.graph.pg_vector_store import PgVectorStore

    rng = np.random.default_rng(11)
    store = PgVectorStore(dim=32, table="vector_test")
    store.clear()
    vectors = rng.standard_normal((40, 32)).astype("float32")
    assert store.add([f"c{i}" for i in range(40)], ["t"] * 40, vectors.tolist()) == 40
    assert store.size == 40
    store.create_index()
    hits = store.search(vectors[5].tolist(), k=3, ef_search=40)
    assert hits[0]["text"] == "c5"
    assert hits[0]["score"] >= 0.999
    store.clear()
    assert store.size == 0


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def test_cache_key_binds_inputs_and_model():
    from service.cache import ResultCache, model_fingerprint

    key_a = ResultCache.key("extract", {"text": "甲"})
    key_b = ResultCache.key("extract", {"text": "乙"})
    assert key_a != key_b
    assert key_a.startswith("medigraph:extract:")
    # Model fingerprint is the last segment: model/backend changes must miss.
    assert key_a.rsplit(":", 1)[1] == model_fingerprint()
    # Key order in the payload must not matter.
    assert ResultCache.key("x", {"a": 1, "b": 2}) == ResultCache.key("x", {"b": 2, "a": 1})


def test_cache_degrades_when_redis_unreachable():
    """A dead cache must mean 'uncached', never an exception or a hang."""
    from service.cache import ResultCache

    dead = ResultCache(url="redis://127.0.0.1:1/0", enabled=True)
    assert dead.get("extract", {"text": "x"}) is None
    dead.set("extract", {"text": "x"}, {"value": 1})  # must not raise
    assert dead.stats()["connected"] is False


def test_cache_disabled_is_a_noop():
    from service.cache import ResultCache

    off = ResultCache(enabled=False)
    off.set("extract", {"text": "x"}, {"value": 1})
    assert off.get("extract", {"text": "x"}) is None


@needs_redis
def test_cache_roundtrip_and_ttl():
    from service.cache import ResultCache

    live = ResultCache(url="redis://127.0.0.1:6380/0", enabled=True, ttl_seconds=60)
    payload = {"text": "缓存往返测试", "backend": "fast"}
    live.set("test", payload, {"entities": [1, 2], "elapsed_ms": 123.0})
    value = live.get("test", payload)
    assert value == {"entities": [1, 2], "elapsed_ms": 123.0}
    assert live.get("test", {"text": "其他"}) is None
