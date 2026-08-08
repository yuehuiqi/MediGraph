"""Offline tests for the AST-level read-only SQL guard.

The bypass cases are the reason this replaced a regex blacklist: each one either
defeats string matching, or was a false positive under it.
"""
from __future__ import annotations

import pytest

from medigraph.analysis.nl2sql import NL2SQL
from medigraph.analysis.sql_guard import ensure_read_only, is_read_only, transpile


# --------------------------------------------------------------------------- #
# accepted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT * FROM patient_visits",
        "SELECT COUNT(*) AS cnt FROM patient_visits WHERE age > 60",
        "SELECT department, AVG(cost) FROM patient_visits GROUP BY department",
        "WITH t AS (SELECT disease, COUNT(*) n FROM patient_visits GROUP BY disease) SELECT * FROM t",
        "SELECT 1 UNION SELECT 2",
        "SELECT 1 INTERSECT SELECT 1",
        "SELECT disease FROM patient_visits EXCEPT SELECT disease FROM patient_visits",
        "SELECT * FROM patient_visits WHERE cost > (SELECT AVG(cost) FROM patient_visits)",
        "SELECT v.disease FROM patient_visits v JOIN lab_tests t ON v.visit_id = t.visit_id",
        "SELECT substr(visit_date,1,7) AS month, COUNT(*) FROM patient_visits GROUP BY month",
        "SELECT 1;",  # single trailing semicolon is fine
    ],
)
def test_legitimate_queries_are_accepted(sql):
    ok, reason = ensure_read_only(sql)
    assert ok, f"{sql!r} rejected: {reason}"


@pytest.mark.parametrize(
    "sql",
    [
        # A regex blacklist flags these on substring match; structurally they are
        # ordinary projections over innocuously-named identifiers.
        "SELECT updated_at FROM patient_visits",
        "SELECT created_at, deleted_flag FROM patient_visits",
        "SELECT drug AS drop_reason FROM prescriptions",
        "SELECT COUNT(*) FROM patient_visits WHERE disease = 'insert-like disease'",
    ],
)
def test_no_false_positives_from_identifier_names(sql):
    ok, reason = ensure_read_only(sql)
    assert ok, f"false positive on {sql!r}: {reason}"


# --------------------------------------------------------------------------- #
# rejected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE patient_visits",
        "DELETE FROM patient_visits",
        "UPDATE patient_visits SET cost = 0",
        "INSERT INTO patient_visits VALUES (1)",
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE patient_visits ADD COLUMN x INT",
        "ATTACH DATABASE 'other.db' AS other",
        "DETACH DATABASE other",
        "PRAGMA table_info(patient_visits)",
        "VACUUM",
        "REINDEX",
    ],
)
def test_write_statements_are_rejected(sql):
    assert not is_read_only(sql)


@pytest.mark.parametrize(
    ("sql", "note"),
    [
        ("SELECT 1; DROP TABLE patient_visits", "second statement"),
        ("SELECT 1;DROP TABLE patient_visits;", "second statement, no space"),
        ("sElEcT 1; dRoP TaBlE patient_visits", "case mixing"),
        ("SELECT 1; -- harmless\nDELETE FROM patient_visits", "comment between statements"),
        ("SELECT /* comment */ 1; UPDATE patient_visits SET cost=0", "inline comment"),
        (
            "WITH x AS (DELETE FROM patient_visits RETURNING 1) SELECT * FROM x",
            "DML nested in a CTE -- a root-only check would miss this",
        ),
        ("SELECT load_extension('evil.so')", "native code loading"),
        ("SELECT readfile('/etc/passwd')", "filesystem read"),
        ("SELECT writefile('out', 'x')", "filesystem write"),
    ],
)
def test_bypass_attempts_are_rejected(sql, note):
    ok, reason = ensure_read_only(sql)
    assert not ok, f"bypass slipped through ({note}): {sql!r}"
    assert reason


@pytest.mark.parametrize("sql", ["", "   ", "not sql at all !!!", None])
def test_empty_and_garbage_are_rejected(sql):
    assert not is_read_only(sql or "")


def test_rejection_reason_is_specific():
    """The reason is fed back to the model for self-correction, so it must say what."""
    _, reason = ensure_read_only("SELECT 1; DROP TABLE t")
    assert "multiple statements" in reason.lower()
    _, reason = ensure_read_only("DELETE FROM t")
    assert "select" in reason.lower() or "delete" in reason.lower()


# --------------------------------------------------------------------------- #
# integration + dialect
# --------------------------------------------------------------------------- #
def test_nl2sql_method_delegates_to_the_guard():
    assert NL2SQL._is_readonly("SELECT 1")
    assert not NL2SQL._is_readonly("SELECT 1; DROP TABLE t")
    assert NL2SQL._is_readonly("SELECT updated_at FROM patient_visits")


def test_transpile_to_postgres_keeps_semantics():
    postgres = transpile(
        "SELECT substr(visit_date,1,7) AS month, COUNT(*) FROM patient_visits GROUP BY month",
        write="postgres",
    )
    assert "SUBSTR" in postgres.upper()
    assert "patient_visits" in postgres
