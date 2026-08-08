"""Offline tests for the NL2SQL deterministic router.

These lock in the fixes found while expanding `benchmarks/nl2sql_hard_natural.json`
from 14 to 44 questions (P3): every one of the router bugs below reproduced as a
genuine 44-question-set failure before the fix (a silently wrong query executed
without error, not a crash), which is why each test asserts the *specific*
correct SQL fragment rather than just "the router deferred" -- a defer alone
would not have caught the original bug (the router did not defer, it silently
answered a different question than the one asked).
"""
from __future__ import annotations

import sqlite3

import pytest

from medigraph.analysis.nl2sql import NL2SQL
from medigraph.analysis.relational import SCHEMA


class _NoLLM:
    """Deterministic-path tests must never reach the network."""

    class _Stats:
        def summary(self) -> dict:
            return {}

    stats = _Stats()

    def chat(self, *args, **kwargs):  # pragma: no cover - guard, not exercised
        raise AssertionError("router test unexpectedly fell through to the LLM")


def _router_db(path) -> None:
    """A richer fixture than test_engineering_upgrade.py's minimal one: two
    departments, two diseases, varied ages/costs/dates, so BETWEEN/exclusion/
    HAVING/superlative-direction tests have more than one row to discriminate
    against.
    """
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    rows = [
        (1, 1, 35, "女", "高血压", "内科", "2024-01-10", 500.0),
        (2, 2, 65, "男", "高血压", "内科", "2024-02-10", 100.0),
        (3, 3, 45, "男", "冠心病", "外科", "2024-03-10", 900.0),
        (4, 4, 72, "女", "冠心病", "外科", "2024-06-10", 300.0),
        (5, 5, 28, "男", "高血压", "外科", "2024-07-10", 700.0),
    ]
    connection.executemany("INSERT INTO patient_visits VALUES(?,?,?,?,?,?,?,?)", rows)
    connection.execute("INSERT INTO prescriptions VALUES(1,1,'阿司匹林',10)")
    connection.execute("INSERT INTO lab_tests VALUES(1,1,'血压',1)")
    connection.execute("INSERT INTO lab_tests VALUES(2,1,'血糖',0)")
    connection.execute("INSERT INTO kg_entities VALUES('高血压','Disease')")
    connection.execute(
        "INSERT INTO kg_triples VALUES('高血压','Disease','recommend_drug','阿司匹林','Drug',1.0,'t')"
    )
    connection.commit()
    connection.close()


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "router.db"
    _router_db(db)
    return NL2SQL(str(db), llm=_NoLLM())


# --------------------------------------------------------------------------- #
# real capability added: two-sided age range
# --------------------------------------------------------------------------- #
def test_age_between_range(engine):
    result = engine.query("30到50岁之间的患者有多少次就诊？")
    assert "BETWEEN 30 AND 50" in result["sql"]
    assert result["generation_mode"] == "deterministic_template"
    assert result["rows"] == [(2,)]  # visits 1 (35) and 3 (45)


def test_age_between_range_handles_reversed_bounds(engine):
    """"50到30岁之间" is nonsensical phrasing but must not emit BETWEEN 50 AND 30
    (which SQLite would silently evaluate to an always-empty range)."""
    result = engine.query("50到30岁之间的患者有多少次就诊？")
    assert "BETWEEN 30 AND 50" in result["sql"]


# --------------------------------------------------------------------------- #
# real capability added: specific test_name shadowed by the generic template
# --------------------------------------------------------------------------- #
def test_specific_test_name_abnormal_count_not_shadowed_by_generic_groupby(engine):
    """Regression: "血压检查的异常次数" used to match the generic "每个检查项目
    的异常次数" GROUP BY template (both contain 异常+检查), silently ignoring
    that a specific test was named and returning every test grouped instead."""
    result = engine.query("血压检查的异常次数是多少？")
    assert "test_name='血压'" in result["sql"]
    assert "GROUP BY" not in result["sql"]
    assert result["rows"] == [(1,)]


def test_generic_abnormal_groupby_still_works_without_a_named_test(engine):
    result = engine.query("每个检查项目的异常次数是多少")
    assert "GROUP BY test_name" in result["sql"]


# --------------------------------------------------------------------------- #
# real bug fixed: superlative direction and default LIMIT
# --------------------------------------------------------------------------- #
def test_superlative_lowest_uses_ascending_order_and_limit_one(engine):
    """Regression: the avg_cost/avg_age group branches used to hard-code DESC
    unconditionally and never applied LIMIT at all, so "最低的科室是哪个"
    returned every department ordered highest-cost-first with no LIMIT."""
    result = engine.query("平均就诊费用最低的科室是哪个？")
    assert "ASC" in result["sql"]
    assert "LIMIT 1" in result["sql"]


def test_superlative_highest_still_descending(engine):
    result = engine.query("平均就诊费用最高的科室是哪个？")
    assert "DESC" in result["sql"]
    assert "LIMIT 1" in result["sql"]


def test_bare_superlative_defaults_to_limit_one_not_five(engine):
    """Regression: "最多"/"最少" were missing from the count-ranking branch's
    LIMIT trigger entirely (only "最高"/"top"/"前" were checked), so a bare
    "哪种疾病的就诊人次最多" returned the full unranked list. Also guards that
    the *default* top_n (5) is not applied to a bare superlative with no
    explicit "前N" -- it must be exactly 1."""
    result = engine.query("哪种疾病的就诊人次最多？")
    assert "LIMIT 1" in result["sql"]
    assert "LIMIT 5" not in result["sql"]


def test_explicit_top_n_is_respected_alongside_superlative(engine):
    """"接诊高血压病人最多的前三个科室" both groups by department *and* filters
    on a specific disease -- exactly the combination the group-ranking template
    cannot express (see test_group_ranking_defers_... above), so this must defer
    rather than answer with LIMIT 3 and the disease filter silently dropped."""
    sql, _ = NL2SQL._deterministic_sql(engine, "接诊高血压病人最多的前三个科室是哪些？")
    assert sql == ""


# --------------------------------------------------------------------------- #
# real bug fixed: group-ranking silently dropping an unrelated-field filter
# --------------------------------------------------------------------------- #
def test_group_ranking_defers_when_a_different_field_is_also_filtered(engine):
    """Regression: "接诊高血压病人最多的前三个科室是哪些" groups by department
    but also names a specific disease; the department-ranking template has no
    WHERE support, so it used to silently drop "高血压" and rank all visits.
    The router must now either defer (empty sql) or -- once deferred -- the
    LLM path applies the filter; either way the disease constraint must never
    be silently dropped from a *matched* deterministic template.
    """
    linked = engine.schema_linker.link("接诊高血压病人最多的前三个科室是哪些？")
    assert linked["values"], "test fixture must resolve a literal disease value"
    sql, _ = NL2SQL._deterministic_sql(engine, "接诊高血压病人最多的前三个科室是哪些？")
    if sql:  # matched a template -> the filter must be present, not dropped
        assert "高血压" in sql


# --------------------------------------------------------------------------- #
# defer conditions: each must route away from the router's naive templates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "就诊人次第二多的疾病是什么？",  # Nth-rank: no LIMIT/OFFSET template exists
        "倒数第二多的疾病是什么？",
        "除了内科以外，就诊人次最多的科室是哪个？",  # exclusion: no NOT-filter template
        "不包括外科，就诊人次最多的科室是哪个？",
        "2024年上半年高血压患者的就诊次数是多少？",  # half-year: no range template
        "2024年下半年的就诊量是多少？",
        "就诊人次超过50次的科室有哪几个？",  # HAVING-as-listing (digit form -- the
        # defer regex itself only needs to be checked against already-normalized
        # text; Chinese-numeral -> digit normalization is covered separately
        # below and in tests/test_numerals.py, not re-tested here).
    ],
)
def test_router_defers_constructs_it_cannot_express(question):
    q = question.lower()
    assert NL2SQL._router_should_defer(q), f"should defer: {question!r}"


def test_having_defer_composes_with_chinese_numeral_normalization(engine):
    """End-to-end: "超过五十次" (Chinese numeral) must still be recognised as the
    HAVING-listing construct once normalize_numerals has converted it to "50次"
    -- this exercises the full _deterministic_sql pipeline, not the regex alone."""
    sql, _ = NL2SQL._deterministic_sql(engine, "就诊人次超过五十次的科室有哪几个？")
    assert sql == ""


@pytest.mark.parametrize(
    "question",
    [
        "60岁以上的患者有多少人次",
        "每个科室的就诊量是多少",
        "平均就诊费用最低的科室是哪个",
        "哪种疾病的就诊人次最多",
    ],
)
def test_router_does_not_defer_expressible_questions(question):
    """The new defer conditions must be specific to the constructs the router
    genuinely cannot express -- not so broad they swallow ordinary questions
    back into the (slower, non-deterministic) LLM path."""
    assert not NL2SQL._router_should_defer(question.lower())


# --------------------------------------------------------------------------- #
# schema documentation fix: kg_triples.relation is an English key
# --------------------------------------------------------------------------- #
def test_schema_text_documents_kg_relation_values_are_english_keys():
    """Regression: the LLM path generated `relation = '并发症'` (the Chinese
    word from the question) for kg-aware questions, matching zero rows because
    the column actually stores English ontology keys like 'complication'."""
    from medigraph.analysis.relational import schema_text

    text = schema_text()
    assert "recommend_drug" in text
    assert "complication" in text
    assert "英文代码" in text or "不是中文" in text


# --------------------------------------------------------------------------- #
# real bugs found by a held-out probe (questions in none of the three eval sets)
#
# The three reported eval sets were also the sets the router was patched
# against, so a fresh probe was written to measure generalisation. It found two
# silent-wrong-answer bugs that every one of the 188 evaluated questions had
# missed; both are locked in below.
# --------------------------------------------------------------------------- #
def test_average_age_survives_an_inserted_verb(engine):
    """Regression: "各科室的平均就诊年龄" matched neither the "平均年龄" group
    branch nor anything above it, fell through to the visit-count branch on the
    bare "多少", and answered with a COUNT(*) ranking -- a different metric
    entirely, executed without error. The 44-question set's one average-age
    question happened to spell "平均年龄" out in a trailing clause, so it passed.
    """
    result = engine.query("各科室的平均就诊年龄是多少")
    assert "AVG(age)" in result["sql"]
    assert "COUNT(*)" not in result["sql"]
    assert "GROUP BY department" in result["sql"]


def test_average_age_with_an_inserted_verb_also_works_as_a_scalar(engine):
    result = engine.query("外科的平均就诊年龄是多少")
    assert "AVG(age)" in result["sql"]
    assert "department='外科'" in result["sql"]


def test_explicit_top_n_counts_visits_not_only_items(engine):
    """Regression: `explicit_top_n` recognised 个/种 but not 次, so "费用最高的
    3次就诊" silently fell back to the default 5 and returned two rows more than
    were asked for. Every ranking question in the eval sets was counted in
    个科室/种药物/种疾病, so 次 was never exercised."""
    result = engine.query("费用最高的3次就诊是哪些患者")
    assert "LIMIT 3" in result["sql"]


@pytest.mark.parametrize(
    "question",
    [
        "就诊人次超过50次的科室有哪几个",
        "开药天数超过7天的处方有多少",
        "60岁以上的患者有多少次就诊",
    ],
)
def test_threshold_numbers_are_not_read_as_a_limit(question):
    """Widening the counter class to 次/位 must not turn a threshold ("超过50
    次") into LIMIT 50. A question carrying a threshold word has no explicit N."""
    from medigraph.analysis.schema_linking import MedicalSchemaLinker

    assert MedicalSchemaLinker.explicit_top_n(question) is None


# --------------------------------------------------------------------------- #
# execution-accuracy comparison: tolerant of extra trailing pred columns
# --------------------------------------------------------------------------- #
def test_rows_match_tolerates_extra_trailing_pred_columns():
    from benchmarks.eval_nl2sql import _rows_match

    assert _rows_match([("内科",)], [("内科", 12)])
    assert _rows_match([("内科", 12)], [("内科", 12)])


def test_rows_match_rejects_pred_with_fewer_columns_than_gold():
    from benchmarks.eval_nl2sql import _rows_match

    assert not _rows_match([("内科", 12)], [("内科",)])


def test_rows_match_still_catches_cardinality_bugs_under_truncation():
    """A join fan-out duplicating rows must still fail even though the
    truncated column shape would otherwise line up."""
    from benchmarks.eval_nl2sql import _rows_match

    assert not _rows_match([("内科",)], [("内科", 12), ("内科", 7)])
