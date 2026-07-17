"""Regression: score_ensemble must not fan a GPU-bound custom metric out across loky worker
processes -- every worker would independently contend for the single physical GPU device
instead of running in parallel. Covers the process-pool analogue of the documented
joblib-threading-over-GPU-work contention bug (CLAUDE.md), gated via
``mlframe.system._gpu_guard.callable_looks_gpu_bound``.
"""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

import mlframe.models.ensembling.score as score_mod
from mlframe.models.ensembling.score import score_ensemble


def _reg_member(seed: int, n: int = 50):
    rng = np.random.default_rng(seed)
    # Positive-only preds: harmonic-mean ("harm") is a sign-sensitive flavour and gets filtered out of
    # ensembling_methods for members with any non-positive prediction, which would collapse the fan-out to a
    # single method and mask the parallel/serial dispatch decision under test.
    return types.SimpleNamespace(
        val_probs=None,
        test_probs=None,
        train_probs=None,
        oof_probs=None,
        val_preds=rng.uniform(0.1, 1.0, size=n),
        test_preds=rng.uniform(0.1, 1.0, size=n),
        train_preds=rng.uniform(0.1, 1.0, size=n),
        oof_preds=None,
    )


def _stub_process_single_ensemble_method(ensemble_method, **kwargs):
    return ensemble_method, np.zeros(2), None


@pytest.fixture(autouse=True)
def _stub_member_processing(monkeypatch):
    # score_ensemble's per-flavour work is orthogonal to the concurrency gate under test; stub it so the
    # test exercises only the parallel-vs-serial dispatch decision, not real ensembling math.
    monkeypatch.setattr(score_mod, "_process_single_ensemble_method", _stub_process_single_ensemble_method)


def _spy_parallel_run(monkeypatch):
    calls = {"n": 0}

    def _fake(tasks, **kw):
        calls["n"] += 1
        return [fn(*args, **kwargs) for fn, args, kwargs in tasks]

    monkeypatch.setattr(score_mod, "parallel_run", _fake)
    return calls


def _gpu_bound_metric(a, b):
    # Never actually invoked (member processing is stubbed); referencing torch by name is enough for the
    # static co_names heuristic to flag this callable as GPU-bound.
    return torch.abs(a - b).mean()  # noqa: F821 - intentionally unresolved, exercises the static heuristic only


def _cpu_metric(a, b):
    return abs(a - b).mean()


def test_gpu_bound_custom_metric_forces_serial(monkeypatch, caplog):
    calls = _spy_parallel_run(monkeypatch)
    members = [_reg_member(0), _reg_member(1)]
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        score_ensemble(
            models_and_predictions=members,
            ensemble_name="t",
            ensembling_methods=["arithm", "harm"],
            custom_ice_metric=_gpu_bound_metric,
            n_jobs=2,
            uncertainty_quantile=0,
            verbose=False,
        )
    assert calls["n"] == 0, "GPU-bound custom_ice_metric must skip the loky process-pool fan-out entirely"
    assert any("GPU-bound" in rec.message for rec in caplog.records)


def test_cpu_only_custom_metric_still_uses_parallel_pool(monkeypatch):
    calls = _spy_parallel_run(monkeypatch)
    members = [_reg_member(0), _reg_member(1)]
    score_ensemble(
        models_and_predictions=members,
        ensemble_name="t",
        ensembling_methods=["arithm", "harm"],
        custom_ice_metric=_cpu_metric,
        n_jobs=2,
        uncertainty_quantile=0,
        verbose=False,
    )
    assert calls["n"] == 1, "a CPU-only custom_ice_metric must keep using the loky process-pool fan-out"


def test_no_custom_metric_still_uses_parallel_pool(monkeypatch):
    calls = _spy_parallel_run(monkeypatch)
    members = [_reg_member(0), _reg_member(1)]
    score_ensemble(
        models_and_predictions=members,
        ensemble_name="t",
        ensembling_methods=["arithm", "harm"],
        n_jobs=2,
        uncertainty_quantile=0,
        verbose=False,
    )
    assert calls["n"] == 1
