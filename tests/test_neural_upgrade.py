"""Tests for the neural-extraction upgrade: shipped result artifacts, the
entity-linking improvements, the multi-hop QA evaluator, and the cascade's
graceful neural fallback.  These run without torch (the heavy model is exercised
by the ML-env benchmarks), so they validate the *evidence* and the integration
contract on any machine.
"""
from __future__ import annotations

import json
from pathlib import Path

from config.settings import ENTITY_LINKER_ARTIFACT, OUTPUTS_DIR
from medigraph.extraction.entity_linker import EntityLinker
from medigraph.graph.local_store import LocalGraphStore


def _load(name: str) -> dict:
    return json.loads((OUTPUTS_DIR / name).read_text(encoding="utf-8"))


def test_neural_dev_eval_artifact_meets_entity_target():
    report = _load("eval_neural_cmeie_dev.json")
    assert report["samples"] >= 3000
    assert report["extractor"] == "neural_gplinker"
    # held-out dev entity F1 clears the 0.75 plan target (macbert-large primary)
    assert report["entity_micro"]["f1"] >= 0.76
    # end-to-end triple F1 on the full 53-relation schema, far above the lexicon
    assert report["end_to_end_triple_micro"]["f1"] >= 0.52


def test_neural_backend_wired_into_operators():
    """Proves the *demo/agent* path (medical_ner/medical_re operators) actually
    routes to the neural GPLinker when it is available -- i.e. benchmark == demo.
    Skips cleanly on machines without torch/the checkpoint (offline CI)."""
    import pytest

    from medigraph.extraction.cascade import load_neural_extractor

    if load_neural_extractor() is None:
        pytest.skip("neural extractor unavailable (no torch/GPU/checkpoint)")

    from medigraph.operators.medical_ner import MedicalNEROperator
    from medigraph.operators.medical_re import MedicalREOperator

    text = "2型糖尿病患者常见多饮、多尿，可用二甲双胍治疗，并发糖尿病肾病。"
    ner = MedicalNEROperator(backend="neural").run({"text": text})
    assert ner["routing"]["level"] == "L1_neural"
    assert any(e.get("extractor") == "neural_gplinker" for e in ner["entities"])
    re = MedicalREOperator(backend="neural").run({"text": text, "entities": ner["entities"]})
    assert any(t.get("extractor") == "neural_gplinker" for t in re.get("triples", []))


def test_cmeie_v1_beats_published_gplinker():
    """On CMeIE-V1 (the dataset the public numbers use), our strict SPO-F1 must
    exceed the comparable published GPLinker (0.598) and CasRel (0.606)."""
    report = _load("eval_neural_cmeie_v1_dev.json")
    assert report["samples"] >= 3000
    strict = report["end_to_end_triple_micro_strict"]["f1"]
    assert strict > 0.606          # exceeds published CasRel / GPLinker
    assert report["entity_micro"]["f1"] >= 0.77
    # strict and lenient must be near-identical (no lenient inflation)
    assert abs(strict - report["end_to_end_triple_micro"]["f1"]) < 0.01


def test_ensemble_lifts_triple_f1():
    ens = _load("eval_ensemble_cmeie_dev.json")
    single = _load("eval_neural_cmeie_dev.json")["end_to_end_triple_micro"]["f1"]
    best = max(v["f1"] for v in ens["triple_micro"].values())
    # the two-encoder ensemble is at least as good as the best single model
    assert best >= single


def test_neural_beats_lexicon_baseline_same_harness():
    neural = _load("eval_neural_cmeie_dev.json")
    lexicon = _load("eval_fast_cmeie_dev.json")
    assert neural["entity_micro"]["f1"] > lexicon["entity_micro"]["f1"]
    assert neural["end_to_end_triple_micro"]["f1"] > lexicon["end_to_end_triple_micro"]["f1"]


def test_entity_linking_artifact_meets_target():
    report = _load("eval_entity_linking.json")
    assert report["overall_accuracy"] >= 0.90
    assert report["nil_rejection_rate"] >= 0.99


def test_scaled_kg_is_self_produced_and_large():
    report = _load("kg_scale_report.json")
    assert report["self_produced"] is True
    assert report["third_party_graph_import"] is False
    assert report["graph"]["num_entities"] >= 30000
    assert report["graph"]["num_triples"] >= 50000


def test_kg_qa_artifact_multihop_and_safety():
    report = _load("eval_kg_qa.json")
    assert report["num_questions"] >= 80
    assert report["multi_hop_accuracy"] >= 0.80
    assert report["provenance_rate"] >= 0.99
    assert report["safe_rejection_rate"] >= 0.90


def test_nl2sql_reports_generation_mode_split():
    """The NL2SQL eval must expose deterministic-router vs LLM split so the
    headline accuracy is not read as pure-LLM."""
    report = _load("eval_nl2sql.json")
    assert "generation_mode_breakdown" in report
    assert sum(v["n"] for v in report["generation_mode_breakdown"].values()) == report["samples"]


def test_real_mention_linking_coverage_after_kb_expansion():
    report = _load("eval_entity_linking_real.json")
    assert report["distinct_mentions"] > 500
    # KB expanded to CM3KG+CMeIE+DIAKG -> real-mention coverage must clear 0.5
    assert report["linked_rate"] >= 0.5
    assert "exact" in report["method_breakdown"]


def test_nl2sql_router_gate_hard_set():
    """The complexity gate must lift the non-template set and keep the template
    set at 100% with no LLM calls.

    Threshold history: the non-template set grew from 14 to 44 genuinely
    diverse questions (P3), which surfaced real router bugs (silently dropped
    BETWEEN ranges, exclusion clauses, HAVING thresholds, a specific-value
    filter shadowed by a generic GROUP BY, and a Unicode-lowercasing bug that
    corrupted Roman-numeral drug names) -- all fixed, not routed around; see
    medigraph/analysis/nl2sql.py's `_router_should_defer` and the superlative
    ordering helper for the fixes and their rationale. Measured accuracy on the
    44-question set is 100% (two independent runs); the assertion below is
    intentionally looser (>= 90%) because natural-language column-shape choices
    ("department" vs "department, count" for "which department has the most
    visits") have observed run-to-run API-level variance at temperature=0 --
    eval_nl2sql.py's `_rows_match` already tolerates a prediction adding extra
    trailing columns beyond what gold asked for, so this threshold is a
    regression guard against real accuracy loss, not against that harmless
    variance.
    """
    hard = _load("eval_nl2sql_nl2sql_hard_natural.json")
    assert hard["samples"] >= 40  # guards against the set being silently shrunk
    assert hard["execution_accuracy"] >= 0.90
    assert hard["dual_database_execution_accuracy"] >= 0.90
    template = _load("eval_nl2sql.json")
    assert template["execution_accuracy"] == 1.0
    modes = template["generation_mode_breakdown"]
    assert modes.get("llm", {}).get("n", 0) == 0  # templates never need the LLM


def test_cmeie_test_predictions_official_format():
    path = OUTPUTS_DIR / "CMeIE_test_pred.jsonl"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4482
    first = json.loads(lines[0])
    assert "text" in first and "spo_list" in first
    if first["spo_list"]:
        spo = first["spo_list"][0]
        assert {"predicate", "subject", "subject_type", "object", "object_type"} <= set(spo)


def test_entity_linker_default_threshold_and_alias():
    linker = EntityLinker.load(ENTITY_LINKER_ARTIFACT)
    assert linker.fuzzy_threshold == 0.85
    # NIL rejection on an obvious non-entity
    res = linker.link("zzz_not_a_medical_entity_123", "")
    assert res["match_method"] == "unlinked_local_id"


def test_cascade_neural_loader_is_optional():
    # On a machine without torch this must return None rather than raise,
    # so the lexicon path keeps the package importable everywhere.
    from medigraph.extraction.cascade import load_neural_extractor

    result = load_neural_extractor("data/models/__does_not_exist__")
    assert result is None


def test_qa_eval_question_builder_on_small_graph():
    from benchmarks.eval_kg_qa import build_questions, evaluate

    store = LocalGraphStore()
    store.upsert_triple("糖尿病", "Disease", "cmeie:clinical_manifestation",
                        "多饮", "Symptom", confidence=0.9, source="unit")
    store.upsert_triple("多饮", "Symptom", "cmeie:related_cause",
                        "高血糖", "Disease", confidence=0.9, source="unit")
    one, two, three = build_questions(store, n_per_hop=1)
    assert one  # at least a 1-hop question is generated
    acc, prov, total, _ = evaluate(store, one, 1)
    assert acc == 1.0 and total >= 1
