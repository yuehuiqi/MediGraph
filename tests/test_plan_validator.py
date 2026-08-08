"""Offline tests for pre-execution plan validation.

The point of the validator is that an unsatisfiable plan costs nothing to reject:
these tests assert it catches the failure *and* that no operator ran.
"""
from __future__ import annotations

import pytest

from medigraph.agents.plan_validator import MAX_NODES, validate_plan
from medigraph.operators.base import BaseOperator, OperatorMeta, register


class _NeedsText(BaseOperator):
    def __init__(self, name="pv_needs_text"):
        self.meta = OperatorMeta(
            name=name,
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {"chunks": {"type": "array"}},
                "required": ["chunks"],
            },
            description="needs text, produces chunks",
        )
        self.calls = 0

    def run(self, inputs: dict, **kwargs) -> dict:
        self.calls += 1
        return {"chunks": []}


class _NeedsChunks(_NeedsText):
    def __init__(self, name="pv_needs_chunks"):
        super().__init__(name)
        self.meta.input_schema = {
            "type": "object",
            "properties": {"chunks": {"type": "array"}},
            "required": ["chunks"],
        }
        self.meta.output_schema = {
            "type": "object",
            "properties": {"entities": {"type": "array"}},
            "required": ["entities"],
        }

    def run(self, inputs: dict, **kwargs) -> dict:
        self.calls += 1
        return {"entities": []}


class _NoRequirements(BaseOperator):
    def __init__(self, name="pv_free"):
        self.meta = OperatorMeta(
            name=name,
            input_schema={"type": "object"},
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            description="no declared requirements",
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        return {"ok": True}


@pytest.fixture(autouse=True)
def _register_operators():
    register(_NeedsText())
    register(_NeedsChunks())
    register(_NoRequirements())


# --------------------------------------------------------------------------- #
# accepted plans
# --------------------------------------------------------------------------- #
def test_satisfiable_chain_is_accepted():
    dag = [
        {"id": "n1", "op": "pv_needs_text", "deps": []},
        {"id": "n2", "op": "pv_needs_chunks", "deps": ["n1"]},
    ]
    report = validate_plan(dag, available={"text"})
    assert report.ok, report.as_dict()
    assert report.layers == [["n1"], ["n2"]]


def test_arg_supplied_input_counts_as_satisfied():
    """A constant in `args` satisfies a requirement the payload lacks."""
    dag = [{"id": "n1", "op": "pv_needs_text", "deps": [], "args": {"text": "inline"}}]
    assert validate_plan(dag, available=set()).ok


def test_operator_without_declared_requirements_is_permitted():
    """Under-annotated operators must degrade to permissive, not block a valid plan."""
    dag = [{"id": "n1", "op": "pv_free", "deps": []}]
    assert validate_plan(dag, available=set()).ok


# --------------------------------------------------------------------------- #
# rejected plans
# --------------------------------------------------------------------------- #
def test_missing_initial_input_is_reported():
    dag = [{"id": "n1", "op": "pv_needs_text", "deps": []}]
    report = validate_plan(dag, available={"paths"})
    assert not report.ok
    assert report.errors[0].code == "unsatisfied_input"
    assert report.errors[0].missing == ["text"]


def test_out_of_order_dependency_is_reported():
    """`pv_needs_chunks` has no upstream producing `chunks`."""
    dag = [{"id": "n1", "op": "pv_needs_chunks", "deps": []}]
    report = validate_plan(dag, available={"text"})
    assert not report.ok
    assert report.errors[0].missing == ["chunks"]


def test_all_errors_are_reported_at_once():
    """One round of feedback must list every problem, not just the first."""
    dag = [
        {"id": "n1", "op": "pv_needs_chunks", "deps": []},
        {"id": "n2", "op": "pv_needs_text", "deps": []},
        {"id": "n3", "op": "does_not_exist", "deps": []},
    ]
    report = validate_plan(dag, available=set())
    codes = {error.code for error in report.errors}
    assert "unsatisfied_input" in codes
    assert "unknown_operator" in codes
    assert len(report.errors) >= 3


def test_peers_in_a_layer_cannot_satisfy_each_other():
    """Sibling outputs merge only at layer end, so a peer cannot be a producer."""
    dag = [
        {"id": "a", "op": "pv_needs_text", "deps": []},          # produces chunks
        {"id": "b", "op": "pv_needs_chunks", "deps": []},        # same layer -> unsatisfied
    ]
    report = validate_plan(dag, available={"text"})
    assert not report.ok
    assert any(error.node == "b" and error.missing == ["chunks"] for error in report.errors)


def test_empty_plan_is_rejected():
    report = validate_plan([], available={"text"})
    assert not report.ok
    assert report.errors[0].code == "empty_plan"


def test_cycle_is_reported_as_structure_error():
    dag = [
        {"id": "a", "op": "pv_free", "deps": ["b"]},
        {"id": "b", "op": "pv_free", "deps": ["a"]},
    ]
    report = validate_plan(dag, available=set())
    assert not report.ok
    assert report.errors[-1].code == "invalid_structure"


def test_node_budget_is_enforced():
    dag = [{"id": f"n{i}", "op": "pv_free", "deps": []} for i in range(MAX_NODES + 5)]
    report = validate_plan(dag, available=set())
    assert not report.ok
    assert any(error.code == "budget_nodes" for error in report.errors)


def test_depth_budget_is_enforced():
    dag = [{"id": "n0", "op": "pv_free", "deps": []}]
    for index in range(1, 20):
        dag.append({"id": f"n{index}", "op": "pv_free", "deps": [f"n{index - 1}"]})
    report = validate_plan(dag, available=set(), max_depth=5)
    assert not report.ok
    assert any(error.code == "budget_depth" for error in report.errors)


# --------------------------------------------------------------------------- #
# the actual guarantee: rejection is free
# --------------------------------------------------------------------------- #
def test_rejected_plan_executes_nothing():
    operator = _NeedsChunks("pv_counted")
    register(operator)
    dag = [{"id": "n1", "op": "pv_counted", "deps": []}]
    report = validate_plan(dag, available={"text"})
    assert not report.ok
    assert operator.calls == 0, "validation must not run the operator"


def test_feedback_names_nodes_and_missing_keys():
    dag = [{"id": "n1", "op": "pv_needs_chunks", "deps": []}]
    feedback = validate_plan(dag, available={"text"}).feedback()
    assert "n1" in feedback
    assert "chunks" in feedback
    assert "unsatisfied_input" in feedback


def test_report_serializes_for_logging():
    dag = [{"id": "n1", "op": "pv_needs_text", "deps": []}]
    data = validate_plan(dag, available={"text"}).as_dict()
    assert data["ok"] is True
    assert data["node_count"] == 1
    assert data["depth"] == 1
    assert data["errors"] == []
