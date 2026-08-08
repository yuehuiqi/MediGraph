"""Offline tests for the L1 cascade, calibration and full schema adapter."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from config.settings import PROJECT_ROOT
from medigraph.extraction.calibration import (
    TemperatureCalibrator,
    expected_calibration_error,
    reliability_bins,
)
from medigraph.extraction.entity_linker import EntityLinker, LinkEntry, stable_entity_id
from medigraph.extraction.fast_path import FastSpanRelationExtractor
from medigraph.schema.cmeie_schema import load_schema, predicate_key

# The official CMeIE-V2 schema file ships under its own licence and therefore
# lives *outside* this repository (see data/external_manifest.json). Tests that
# need it are skipped when the raw dataset has not been placed next to the repo,
# so a fresh clone and CI both stay green.
CMEIE_V2_SCHEMA = PROJECT_ROOT.parent / "CMeIE-V2" / "53_schemas.json"
requires_cmeie_v2_schema = pytest.mark.skipif(
    not CMEIE_V2_SCHEMA.exists(),
    reason=(
        f"external dataset missing: {CMEIE_V2_SCHEMA}. "
        "Place the raw CMeIE-V2 release one level above the repo root "
        "(see data/external_manifest.json) to enable this check."
    ),
)


@pytest.mark.parametrize("value", [0.01, 0.2, 0.5, 0.8, 0.99])
def test_calibration_transform_is_probability(value):
    calibrated = TemperatureCalibrator(temperature=1.2, bias=-0.3).transform_one(value)
    assert 0.0 < calibrated < 1.0


def test_calibration_reduces_synthetic_ece():
    confidences = [0.9] * 10 + [0.7] * 10
    labels = [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8
    before = expected_calibration_error(confidences, labels)
    model = TemperatureCalibrator().fit(confidences, labels)
    after = expected_calibration_error(model.transform(confidences), labels)
    assert after < before


def test_reliability_bins_keep_empty_bins():
    bins = reliability_bins([0.1, 0.9], [0, 1], n_bins=5)
    assert len(bins) == 5
    assert sum(bucket["count"] for bucket in bins) == 2


def test_calibrator_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    model = TemperatureCalibrator(temperature=1.4, bias=-0.2, fitted_samples=12)
    model.save(path)
    loaded = TemperatureCalibrator.load(path)
    assert loaded.temperature == 1.4
    assert loaded.bias == -0.2
    assert loaded.fitted_samples == 12


@pytest.mark.parametrize(
    ("name", "entity_type"),
    [("高血压", "Disease"), ("二甲双胍", "Drug"), ("HER2", "Biomarker")],
)
def test_stable_entity_id_is_deterministic(name, entity_type):
    assert stable_entity_id(name, entity_type) == stable_entity_id(name, entity_type)
    assert stable_entity_id(name, entity_type).startswith("MEDIGRAPH:")


def _linker() -> EntityLinker:
    return EntityLinker(
        [
            LinkEntry("D1", "糖尿病", "Disease", aliases=["diabetes"], source="test"),
            LinkEntry("M1", "阿司匹林", "Drug", source="test"),
        ],
        fuzzy_threshold=0.65,
    )


@pytest.mark.parametrize(
    ("surface", "method", "canonical_id"),
    [
        ("糖尿病", "exact", "D1"),
        ("diabetes", "exact", "D1"),
        ("阿斯匹林", "alias", "M1"),
        ("糖尿并", "fuzzy", "D1"),
    ],
)
def test_entity_linker_match_modes(surface, method, canonical_id):
    result = _linker().link(surface)
    assert result["match_method"] == method
    assert result["canonical_id"] == canonical_id


def test_entity_linker_unlinked_gets_local_id():
    result = _linker().link("全新概念", "Other")
    assert result["match_method"] == "unlinked_local_id"
    assert result["canonical_id"].startswith("MEDIGRAPH:")


@pytest.fixture
def tiny_extractor() -> FastSpanRelationExtractor:
    artifact = {
        "version": "test",
        "entities": [
            {"name": "2型糖尿病", "type": "Disease", "count": 20, "source": "gold"},
            {"name": "糖尿病", "type": "Disease", "count": 15, "source": "gold"},
            {"name": "多饮", "type": "Symptom", "count": 8, "source": "gold"},
            {"name": "二甲双胍", "type": "Drug", "count": 12, "source": "gold"},
            {"name": "二甲双胍", "type": "Disease", "count": 1, "source": "noise"},
        ],
        "relations": [
            {
                "head": "2型糖尿病",
                "relation": "recommend_drug",
                "tail": "二甲双胍",
                "count": 4,
                "source": "gold",
            }
        ],
        "cmeie_benchmark_relations": [
            {
                "head": "2型糖尿病",
                "head_type": "Disease",
                "relation": "cmeie:drug_treatment",
                "predicate": "药物治疗",
                "tail": "二甲双胍",
                "tail_type": "Drug",
                "count": 4,
            }
        ],
    }
    return FastSpanRelationExtractor(artifact)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2型糖尿病", "2型糖尿病"),
        ("患者使用二甲双胍", "二甲双胍"),
        ("常有多饮表现", "多饮"),
        ("无医学术语", None),
    ],
)
def test_fast_span_extraction(tiny_extractor, text, expected):
    names = [item["name"] for item in tiny_extractor.extract_entities(text)]
    assert (expected in names) if expected else not names


def test_fast_span_type_conflict_resolution(tiny_extractor):
    entities = tiny_extractor.extract_entities("二甲双胍")
    assert [item["type"] for item in entities] == ["Drug"]
    alternatives = tiny_extractor.extract_entities("二甲双胍", include_type_alternatives=True)
    assert {item["type"] for item in alternatives} == {"Drug", "Disease"}


def test_fast_span_overlap_policy(tiny_extractor):
    nested = tiny_extractor.extract_entities("2型糖尿病")
    maximal = tiny_extractor.extract_entities("2型糖尿病", overlap_policy="maximal")
    assert len(nested) == 2
    assert [item["name"] for item in maximal] == ["2型糖尿病"]


def test_fast_relation_memorized_fact(tiny_extractor):
    text = "2型糖尿病可用二甲双胍治疗。"
    entities = tiny_extractor.extract_entities(text)
    triples = tiny_extractor.extract_relations(text, entities)
    assert any(item["relation"] == "recommend_drug" for item in triples)
    assert triples[0]["evidence_method"] == "memorized_fact"


def test_fast_cmeie_full_label_output(tiny_extractor):
    triples = tiny_extractor.extract_cmeie_relations("2型糖尿病可用二甲双胍。")
    assert triples[0]["predicate"] == "药物治疗"
    assert triples[0]["relation"] == "cmeie:drug_treatment"


def test_uncertainty_empty_prediction(tiny_extractor):
    result = tiny_extractor.uncertainty([])
    assert result["uncertain"]
    assert result["reason"] == "no_prediction"


@requires_cmeie_v2_schema
def test_cmeie_schema_has_all_53_rows():
    rows = load_schema(CMEIE_V2_SCHEMA)
    assert len(rows) == 53
    assert all(row["predicate_key"] for row in rows)


def test_cmeie_predicate_keys_cover_all_44_labels():
    """Repo-local guard for the same invariant.

    The 53 official schema rows use 44 distinct predicate labels; this asserts the
    bundled mapping stays complete and collision-free without needing the raw
    dataset, so the property is still covered on a fresh clone.
    """
    from medigraph.schema.cmeie_schema import CMEIE_PREDICATE_KEYS

    assert len(CMEIE_PREDICATE_KEYS) == 44
    assert len(set(CMEIE_PREDICATE_KEYS.values())) == 44
    assert all(predicate_key(label) for label in CMEIE_PREDICATE_KEYS)


@pytest.mark.parametrize(
    ("predicate", "key"),
    [("药物治疗", "cmeie:drug_treatment"), ("临床表现", "cmeie:clinical_manifestation")],
)
def test_cmeie_predicate_key(predicate, key):
    assert predicate_key(predicate) == key
