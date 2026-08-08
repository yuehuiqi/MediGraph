"""Offline tests for Chinese-numeral normalization in the NL2SQL path.

Covers the parser, the two gates (unit / rank prefix), the vague-quantifier guard
and the vocabulary mask that keeps medical terminology intact.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from config.settings import OUTPUTS_DIR
from medigraph.analysis.numerals import (
    chinese_to_int,
    normalize,
    normalize_numerals,
    numeral_bearing,
)

# Terms that a naive rewrite corrupts; all are real values from the analytics DB.
MEDICAL_TERMS = [
    "十二指肠白点综合征",
    "二十五味松石丸",
    "血液生化六项检查",
    "一秒用力呼出量／用力肺活量比值",
    "百日咳",
    "复方万年青胶囊",
    "二天油",
    "肝纤四项",
    "三尖瓣闭锁",
    "一氧化碳中毒",
    "第三脑室肿瘤",
    "四肢瘫痪",
    "香砂六君丸",
    "十全大补膏",
]


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一", 1), ("七", 7), ("九", 9),
        ("十", 10), ("十五", 15), ("二十", 20), ("二十五", 25),
        ("六十", 60), ("九十九", 99),
        ("一百", 100), ("一百零八", 108), ("一百二十", 120), ("三百五十六", 356),
        ("一千", 1000), ("三千五百", 3500),
        ("一万", 10_000), ("两万", 20_000),
        ("两", 2), ("俩", 2),
        ("零", 0), ("〇", 0),
    ],
)
def test_chinese_to_int(text, expected):
    assert chinese_to_int(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "岁", "第", "3"])
def test_chinese_to_int_rejects_non_numerals(text):
    assert chinese_to_int(text) is None


# --------------------------------------------------------------------------- #
# unit gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("六十岁以上高血压患者的平均住院费用", "60岁以上高血压患者的平均住院费用"),
        ("五十岁以下男性的平均花费", "50岁以下男性的平均花费"),
        ("七十五岁以上的患者有多少人次", "75岁以上的患者有多少人次"),
        ("开药天数超过七天的处方", "开药天数超过7天的处方"),
        ("做过两种及以上检验项目的就诊", "做过2种及以上检验项目的就诊"),
        ("前三个科室", "前3个科室"),
        ("前十名患者", "前10名患者"),
    ],
)
def test_unit_and_rank_gates_rewrite(question, expected):
    assert normalize(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "十二指肠溃疡的患者有多少",   # 指 is not a counter
        "三叉神经痛怎么治",           # 叉 is not a counter
        "二尖瓣狭窄的用药",           # 尖 is not a counter
        "一氧化碳中毒的病例数",       # 氧 is not a counter
    ],
)
def test_non_counter_context_is_left_alone(question):
    assert normalize(question) == question


# --------------------------------------------------------------------------- #
# vague-quantifier guard: rewriting these would reintroduce silent wrong answers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    ["十几岁的患者有多少人次", "几十个科室", "二十几岁的病人", "十几个不同的药"],
)
def test_vague_quantifiers_are_not_rewritten(question):
    assert normalize(question) == question


# --------------------------------------------------------------------------- #
# vocabulary mask
# --------------------------------------------------------------------------- #
def test_vocabulary_mask_protects_medical_terms():
    protected = numeral_bearing(MEDICAL_TERMS)
    for term in MEDICAL_TERMS:
        question = f"{term}的患者有多少人次"
        assert normalize(question, protected) == question, term


def test_mask_is_actually_required():
    """Without the mask, ambiguous counters (味/项/年/天/日) do corrupt terms.

    Guards against someone "simplifying" the linker by dropping the protected set.
    """
    unmasked = [term for term in MEDICAL_TERMS if normalize(term) != term]
    assert unmasked, "expected some terms to be corrupted without the mask"


def test_mask_still_rewrites_the_quantity_in_the_same_question():
    protected = numeral_bearing(MEDICAL_TERMS)
    normalized, rewrites = normalize_numerals(
        "血液生化六项检查中六十岁以上的患者有多少", protected
    )
    assert "血液生化六项检查" in normalized
    assert "60岁以上" in normalized
    assert [r["value"] for r in rewrites] == [60]


def test_rewrite_log_records_reason():
    _, rewrites = normalize_numerals("六十岁以上")
    assert rewrites == [{"surface": "六十", "value": 60, "reason": "unit:岁"}]


def test_normalize_is_idempotent_on_ascii():
    for question in ["60岁以上的患者", "前3个科室", "SELECT 1"]:
        assert normalize(question) == question


# --------------------------------------------------------------------------- #
# end-to-end regression against the real vocabulary
# --------------------------------------------------------------------------- #
ANALYTICS_DB = Path(OUTPUTS_DIR) / "analytics.db"


@pytest.mark.skipif(not ANALYTICS_DB.exists(), reason="analytics.db not built yet")
def test_no_db_vocabulary_value_is_corrupted():
    """The mask must keep every one of the ~7.5k real values byte-identical."""
    connection = sqlite3.connect(f"file:{ANALYTICS_DB}?mode=ro", uri=True)
    values: set[str] = set()
    for sql in (
        "SELECT DISTINCT disease FROM patient_visits",
        "SELECT DISTINCT department FROM patient_visits",
        "SELECT DISTINCT drug FROM prescriptions",
        "SELECT DISTINCT test_name FROM lab_tests",
        "SELECT DISTINCT name FROM kg_entities",
    ):
        values |= {str(row[0]) for row in connection.execute(sql) if row[0]}
    connection.close()
    protected = numeral_bearing(values)
    corrupted = [value for value in values if normalize(value, protected) != value]
    assert not corrupted, corrupted[:10]


@pytest.mark.skipif(not ANALYTICS_DB.exists(), reason="analytics.db not built yet")
def test_link_omits_empty_numeral_log():
    """`link()` output is embedded verbatim in the LLM prompt.

    An always-present `numerals` key would perturb every prompt (and re-roll the
    model's output) even for questions with no numerals, so the key must be absent
    when nothing was rewritten. This regression was observed as an unrelated
    NL2SQL case flipping after the key was added unconditionally.
    """
    from medigraph.analysis.schema_linking import MedicalSchemaLinker

    linker = MedicalSchemaLinker(str(ANALYTICS_DB))
    assert "numerals" not in linker.link("哪个科室接诊的不同病人数量最多？")
    assert linker.link("六十岁以上的患者")["numerals"]


@pytest.mark.skipif(not ANALYTICS_DB.exists(), reason="analytics.db not built yet")
def test_deterministic_router_keeps_chinese_age_filter():
    """The regression this module exists for: the age predicate must survive."""
    from medigraph.analysis.nl2sql import NL2SQL
    from medigraph.analysis.schema_linking import MedicalSchemaLinker

    engine = NL2SQL.__new__(NL2SQL)
    engine.schema_linker = MedicalSchemaLinker(str(ANALYTICS_DB))

    chinese, _ = NL2SQL._deterministic_sql(engine, "六十岁以上高血压患者的平均住院费用是多少？")
    arabic, _ = NL2SQL._deterministic_sql(engine, "60岁以上高血压患者的平均住院费用是多少？")
    assert "age>60" in chinese.replace(" ", "")
    assert chinese == arabic
