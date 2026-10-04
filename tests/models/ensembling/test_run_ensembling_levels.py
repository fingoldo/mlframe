"""Behaviour of the per-level ensembling loop: level chaining, conf entries, parallel dispatch and the sequential fallbacks."""

from __future__ import annotations

import logging

import pytest

from mlframe.models.ensembling import score_levels
from mlframe.models.ensembling.score_levels import run_ensembling_levels


class _Recorder:
    """Fake ``process_fn`` recording the member list it was given at every (level, method) call."""

    def __init__(self, conf_for=()):
        """Store which methods return a confidence result."""
        self.calls = []
        self.conf_for = set(conf_for)

    def __call__(self, *, ensemble_method, level_models_and_predictions, ensembling_level, **kwargs):
        """Record the call and return ``(method, next members, conf)``."""
        self.calls.append((ensembling_level, ensemble_method, list(level_models_and_predictions), kwargs))
        out = f"{ensemble_method}@L{ensembling_level}"
        return ensemble_method, out, (f"conf-{out}" if ensemble_method in self.conf_for else None)


def _must_not_run(*args, **kwargs):
    """Parallel runner that fails the test if the loop dispatches in parallel."""
    raise AssertionError("parallel_run_fn must not be used")


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch):
    """Reset the shared log throttle so the fallback warnings are not swallowed by earlier tests."""
    import importlib

    monkeypatch.setattr(importlib.import_module("mlframe.utils.log_throttle"), "_counts", {})


def _run(process_fn, parallel_run_fn=_must_not_run, methods=("arithm", "harm"), levels=2, n_jobs=1, base_params=None, members=("m0", "m1")):
    """Invoke the loop and return the filled ``res`` dict."""
    res: dict = {}
    run_ensembling_levels(
        res=res,
        level_models_and_predictions=list(members),
        ensembling_methods=list(methods),
        max_ensembling_level=levels,
        effective_n_jobs=n_jobs,
        base_params=base_params if base_params is not None else {"is_regression": True},
        process_fn=process_fn,
        parallel_run_fn=parallel_run_fn,
    )
    return res


def test_sequential_levels_chain_each_levels_outputs_into_the_next_as_members():
    """Level 0 sees the original members; level 1 sees exactly level 0's per-method results, in method order."""
    rec = _Recorder()
    res = _run(rec, levels=2)
    level0 = [c for c in rec.calls if c[0] == 0]
    level1 = [c for c in rec.calls if c[0] == 1]
    assert [c[2] for c in level0] == [["m0", "m1"], ["m0", "m1"]]
    assert [c[2] for c in level1] == [["arithm@L0", "harm@L0"], ["arithm@L0", "harm@L0"]]
    assert res == {"arithm": "arithm@L1", "harm": "harm@L1"}


def test_base_params_reach_process_fn_and_conf_entries_are_stored_only_when_returned():
    """Level-invariant kwargs are forwarded verbatim; ``"<method> conf"`` exists only for methods that returned a conf."""
    rec = _Recorder(conf_for={"harm"})
    res = _run(rec, levels=1, base_params={"is_regression": False, "nbins": 7})
    assert all(c[3] == {"is_regression": False, "nbins": 7} for c in rec.calls)
    assert res["harm conf"] == "conf-harm@L0"
    assert "arithm conf" not in res


def test_zero_levels_leaves_res_untouched():
    """max_ensembling_level=0 runs nothing."""
    rec = _Recorder()
    assert _run(rec, levels=0) == {}
    assert rec.calls == []


def test_single_method_never_uses_the_parallel_runner_even_with_many_jobs():
    """Parallel dispatch needs more than one method."""
    rec = _Recorder()
    res = _run(rec, methods=("arithm",), n_jobs=4)
    assert res == {"arithm": "arithm@L1"}


def test_parallel_dispatch_uses_loky_and_collects_every_methods_result():
    """With several methods and n_jobs>1 the tasks go through parallel_run_fn(loky, n_jobs) and results are chained."""
    rec = _Recorder(conf_for={"arithm"})
    seen = []

    def fake_parallel(tasks, **kwargs):
        """Run joblib ``delayed`` tuples inline and record the dispatch options."""
        seen.append(kwargs)
        return [func(*args, **kw) for func, args, kw in tasks]

    res = _run(rec, parallel_run_fn=fake_parallel, n_jobs=3, levels=2)
    assert len(seen) == 2
    assert all(k["n_jobs"] == 3 and k["backend"] == "loky" and k["max_nbytes"] == "1K" for k in seen)
    assert res["arithm"] == "arithm@L1" and res["arithm conf"] == "conf-arithm@L1"
    assert [c[2] for c in rec.calls if c[0] == 1] == [["arithm@L0", "harm@L0"]] * 2


def test_unpicklable_custom_metric_falls_back_to_sequential_with_warning_for_all_levels(caplog):
    """A lambda metric cannot cross the loky boundary, so the loop warns once and the fallback sticks for the remaining levels."""
    rec = _Recorder()
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        res = _run(rec, parallel_run_fn=_must_not_run, n_jobs=4, levels=2, base_params={"custom_ice_metric": lambda *a: 0.0})
    assert res == {"arithm": "arithm@L1", "harm": "harm@L1"}
    assert len(rec.calls) == 4
    assert any("not picklable" in r.getMessage() for r in caplog.records)


def test_gpu_bound_custom_metric_falls_back_to_sequential_with_warning(monkeypatch, caplog):
    """A metric that looks GPU-bound is never fanned out across worker processes."""
    monkeypatch.setattr(score_levels, "callable_looks_gpu_bound", lambda fn: fn is not None)
    rec = _Recorder()
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        res = _run(rec, parallel_run_fn=_must_not_run, n_jobs=4, levels=1, base_params={"custom_ice_metric": _must_not_run})
    assert res == {"arithm": "arithm@L0", "harm": "harm@L0"}
    assert any("GPU-bound" in r.getMessage() for r in caplog.records)
