"""Offline tests for layered parallel DAG execution and node timeouts.

Covers the three properties the layered executor has to guarantee:
  * layering is correct (independent nodes share a layer, dependents do not);
  * a layer really runs concurrently (wall clock, not sum of sleeps);
  * concurrent nodes cannot clobber each other's payload writes.
Plus the timeout path, including its interaction with fallback.
"""
from __future__ import annotations

import threading
import time

import pytest

from medigraph.agents.dag_executor import (
    DAGExecutor,
    NodeStatus,
    topological_layers,
    topological_order,
)
from medigraph.operators.base import BaseOperator, OperatorMeta, register


class _SleepOperator(BaseOperator):
    """Sleeps, then writes a per-node key. Records observed concurrency."""

    active = 0
    peak = 0
    _lock = threading.Lock()

    def __init__(self, name: str, seconds: float = 0.3, out_key: str = "ok"):
        self.seconds = seconds
        self.out_key = out_key
        self.meta = OperatorMeta(
            name=name,
            input_schema={"type": "object"},
            output_schema={
                "type": "object",
                "properties": {out_key: {"type": "string"}},
                "required": [out_key],
            },
            description="sleep test operator",
        )

    @classmethod
    def reset(cls) -> None:
        cls.active = 0
        cls.peak = 0

    def run(self, inputs: dict, **kwargs) -> dict:
        with _SleepOperator._lock:
            _SleepOperator.active += 1
            _SleepOperator.peak = max(_SleepOperator.peak, _SleepOperator.active)
        try:
            time.sleep(self.seconds)
            return {self.out_key: self.meta.name}
        finally:
            with _SleepOperator._lock:
                _SleepOperator.active -= 1


class _HangOperator(BaseOperator):
    def __init__(self, name: str, seconds: float = 30.0):
        self.seconds = seconds
        self.meta = OperatorMeta(
            name=name,
            input_schema={"type": "object"},
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            description="hangs",
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        time.sleep(self.seconds)
        return {"ok": True}


class _QuickOperator(BaseOperator):
    def __init__(self, name: str):
        self.meta = OperatorMeta(
            name=name,
            input_schema={"type": "object"},
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            description="quick",
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        return {"ok": True}


# --------------------------------------------------------------------------- #
# layering
# --------------------------------------------------------------------------- #
def test_layers_group_independent_nodes():
    dag = [
        {"id": "a", "op": "x", "deps": []},
        {"id": "b", "op": "x", "deps": ["a"]},
        {"id": "c", "op": "x", "deps": ["a"]},
        {"id": "d", "op": "x", "deps": ["b", "c"]},
    ]
    assert topological_layers(dag) == [["a"], ["b", "c"], ["d"]]


def test_layers_of_a_linear_dag_are_all_width_one():
    dag = [
        {"id": "n1", "op": "x", "deps": []},
        {"id": "n2", "op": "x", "deps": ["n1"]},
        {"id": "n3", "op": "x", "deps": ["n2"]},
    ]
    assert topological_layers(dag) == [["n1"], ["n2"], ["n3"]]


def test_layers_flatten_to_a_valid_topological_order():
    dag = [
        {"id": "a", "op": "x", "deps": []},
        {"id": "b", "op": "x", "deps": []},
        {"id": "c", "op": "x", "deps": ["a", "b"]},
    ]
    flat = [nid for layer in topological_layers(dag) for nid in layer]
    assert sorted(flat) == sorted(topological_order(dag))
    assert flat.index("c") > max(flat.index("a"), flat.index("b"))


@pytest.mark.parametrize(
    "dag",
    [
        [{"id": "a", "op": "x", "deps": ["b"]}, {"id": "b", "op": "x", "deps": ["a"]}],  # cycle
        [{"id": "a", "op": "x", "deps": ["ghost"]}],  # unknown dep
        [{"id": "", "op": "x", "deps": []}],  # empty id
        [{"id": "a", "op": "x", "deps": []}, {"id": "a", "op": "x", "deps": []}],  # dup id
    ],
)
def test_layers_reject_invalid_dags(dag):
    with pytest.raises(ValueError):
        topological_layers(dag)


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #
def test_layer_runs_concurrently():
    """Four 0.3s nodes in one layer must finish in well under 4x0.3s."""
    for index in range(4):
        register(_SleepOperator(f"test_par_{index}", seconds=0.3, out_key=f"k{index}"))
    dag = [{"id": f"n{i}", "op": f"test_par_{i}", "deps": []} for i in range(4)]
    _SleepOperator.reset()

    started = time.perf_counter()
    result = DAGExecutor(max_retries=0, max_workers=4).run(dag, {}, verbose=False)
    elapsed = time.perf_counter() - started

    assert all(state["status"] == NodeStatus.SUCCESS for state in result["states"].values())
    assert _SleepOperator.peak > 1, "no two operators ever overlapped"
    assert elapsed < 0.9, f"layer did not run in parallel (took {elapsed:.2f}s)"
    assert result["report"]["max_layer_width"] == 4


def test_serial_mode_is_still_available():
    for index in range(3):
        register(_SleepOperator(f"test_ser_{index}", seconds=0.05, out_key=f"s{index}"))
    dag = [{"id": f"n{i}", "op": f"test_ser_{i}", "deps": []} for i in range(3)]
    _SleepOperator.reset()
    result = DAGExecutor(max_retries=0, parallel=False).run(dag, {}, verbose=False)
    assert _SleepOperator.peak == 1
    assert all(state["status"] == NodeStatus.SUCCESS for state in result["states"].values())


def test_concurrent_nodes_do_not_lose_each_others_output():
    """Each node writes a distinct key; all must survive the layer-end merge."""
    for index in range(4):
        register(_SleepOperator(f"test_merge_{index}", seconds=0.05, out_key=f"key{index}"))
    dag = [{"id": f"n{i}", "op": f"test_merge_{i}", "deps": []} for i in range(4)]
    result = DAGExecutor(max_retries=0, max_workers=4).run(dag, {}, verbose=False)
    for index in range(4):
        assert result["payload"][f"key{index}"] == f"test_merge_{index}"


def test_layer_snapshot_isolation():
    """Nodes in one layer read the payload as it was at layer start.

    Guards the "collect in layer, merge at layer end" rule: if merges happened
    eagerly, a sibling's output would leak into a peer's inputs and execution would
    depend on completion order.
    """
    seen: list[bool] = []

    class _Observer(BaseOperator):
        def __init__(self, name: str):
            self.meta = OperatorMeta(
                name=name,
                input_schema={"type": "object"},
                output_schema={"type": "object", "properties": {"mark": {"type": "string"}}, "required": ["mark"]},
                description="observer",
            )

        def run(self, inputs: dict, **kwargs) -> dict:
            seen.append("mark" in inputs)
            time.sleep(0.05)
            return {"mark": self.meta.name}

    register(_Observer("test_obs_a"))
    register(_Observer("test_obs_b"))
    dag = [
        {"id": "a", "op": "test_obs_a", "deps": []},
        {"id": "b", "op": "test_obs_b", "deps": []},
    ]
    DAGExecutor(max_retries=0, max_workers=2).run(dag, {}, verbose=False)
    assert seen == [False, False], "a sibling's output leaked into a peer's inputs"


# --------------------------------------------------------------------------- #
# timeout
# --------------------------------------------------------------------------- #
def test_node_timeout_marks_failure_instead_of_blocking_forever():
    register(_HangOperator("test_hang", seconds=30.0))
    dag = [{"id": "a", "op": "test_hang", "deps": [], "timeout": 0.2}]
    started = time.perf_counter()
    result = DAGExecutor(max_retries=0).run(dag, {}, verbose=False)
    elapsed = time.perf_counter() - started
    assert result["states"]["a"]["status"] == NodeStatus.FAILED
    assert "timeout" in result["states"]["a"]["error"].lower()
    assert elapsed < 5, f"executor waited {elapsed:.1f}s despite a 0.2s timeout"


def test_executor_default_timeout_applies():
    register(_HangOperator("test_hang_default", seconds=30.0))
    dag = [{"id": "a", "op": "test_hang_default", "deps": []}]
    result = DAGExecutor(max_retries=0, timeout=0.2).run(dag, {}, verbose=False)
    assert result["states"]["a"]["status"] == NodeStatus.FAILED
    assert "timeout" in result["states"]["a"]["error"].lower()


def test_node_timeout_falls_back_to_another_operator():
    """A timeout must enter the same recovery path as any other failure."""
    register(_HangOperator("test_hang_fb", seconds=30.0))
    register(_QuickOperator("test_quick_fb"))
    dag = [
        {
            "id": "a",
            "op": "test_hang_fb",
            "deps": [],
            "timeout": 0.2,
            "on_error": {"fallback_op": "test_quick_fb"},
        }
    ]
    result = DAGExecutor(max_retries=0).run(dag, {}, verbose=False)
    assert result["states"]["a"]["status"] == NodeStatus.SUCCESS
    assert result["recoveries"][0]["strategy"] == "fallback_operator"


def test_timeout_of_zero_means_unbounded():
    register(_QuickOperator("test_quick_unbounded"))
    dag = [{"id": "a", "op": "test_quick_unbounded", "deps": [], "timeout": 0}]
    result = DAGExecutor(max_retries=0).run(dag, {}, verbose=False)
    assert result["states"]["a"]["status"] == NodeStatus.SUCCESS
