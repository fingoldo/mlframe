"""A failed kept stage must leave the state of the stages BEFORE it on the estimator, as a sequential run does (AUD23-06)."""

from __future__ import annotations

import contextlib
import types

import pytest

import mlframe.feature_selection.filters._fe_device_scheduler as sched
import mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency as C


class _Est:
    """A bare estimator: a list of engineered recipes."""

    def __init__(self) -> None:
        self.recipes: list = []


@pytest.fixture
def no_device(monkeypatch):
    """Three fake slots and no real device work."""
    monkeypatch.setattr(sched, "plan_stage_slots", lambda weights, vram_weights=None: [types.SimpleNamespace(device_id=0, budget_bytes=1 << 30) for _ in weights])
    monkeypatch.setattr(sched, "stage_device_slot", lambda slot, i: contextlib.nullcontext())
    monkeypatch.setattr(C, "_synchronize_device", lambda device_id: None)
    monkeypatch.setattr(C, "_stages_fit_device", lambda vram_weights, n_rows: True)


def _stage(name, fail=False):
    """A stage that appends its name to the recipes of its estimator copy, then optionally fails."""

    def run(est):
        """Run."""
        est.recipes.append(name)
        if fail:
            raise RuntimeError(f"stage {name} failed")
        return name

    return run


def test_the_stages_before_the_failing_one_are_merged(no_device):
    """Stage 0 succeeded, stage 1 failed, stage 2 succeeded: the error is raised and the estimator holds stage 0's recipe, not stage 2's (a sequential run never reached it)."""
    est = _Est()
    with pytest.raises(RuntimeError, match="stage b failed"):
        C.run_stages_concurrently(est, [_stage("a"), _stage("b", fail=True), _stage("c")], weights=(1.0, 1.0, 1.0), vram_weights=(1.0, 1.0, 1.0))
    assert est.recipes == ["a"]


def test_a_failure_in_the_first_stage_merges_nothing(no_device):
    """Nothing precedes the first stage: the estimator is untouched."""
    est = _Est()
    with pytest.raises(RuntimeError, match="stage a failed"):
        C.run_stages_concurrently(est, [_stage("a", fail=True), _stage("b"), _stage("c")], weights=(1.0, 1.0, 1.0), vram_weights=(1.0, 1.0, 1.0))
    assert est.recipes == []


def test_all_stages_succeeding_still_merges_all(no_device):
    """The normal path is unchanged."""
    est = _Est()
    out = C.run_stages_concurrently(est, [_stage("a"), _stage("b"), _stage("c")], weights=(1.0, 1.0, 1.0), vram_weights=(1.0, 1.0, 1.0))
    assert out == ["a", "b", "c"]
    assert est.recipes == ["a", "b", "c"]
