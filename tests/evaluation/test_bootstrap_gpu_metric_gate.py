"""Regression: bootstrap_metrics must not parallelise the resample loop when a user-supplied metric_fns /
metric_fns_idx callable looks GPU-bound, regardless of MLFRAME_BOOTSTRAP_BACKEND. Fanning a GPU-bound metric out
across threads or processes makes every worker independently contend for the single physical GPU device instead
of parallelising -- the same failure mode the joblib-threading-over-GPU-work CLAUDE.md entry documents.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.evaluation.bootstrap import bootstrap_metrics

_N = 8000  # above the n>=5000 parallel gate
_R = 400  # above the n_bootstrap>=256 gate


def _data(seed=0):
    """Helper that data."""
    rng = np.random.default_rng(seed)
    y = (rng.random(_N) < 0.35).astype(np.float64)
    p = np.clip(0.2 + 0.5 * y + rng.standard_normal(_N) * 0.3, 1e-6, 1 - 1e-6)
    return y, p


def _gpu_bound_metric(yy, pp):
    # The point-estimate call happens unconditionally (bootstrap_metrics computes it before the parallel gate),
    # so the dead branch keeps the callable runnable while still leaving 'torch' in its co_names for the static
    # callable_looks_gpu_bound heuristic to detect.
    """Helper that gpu bound metric."""
    if False:
        torch.zeros(1)  # noqa: F821
    return float(np.mean((yy - pp) ** 2))


def _cpu_metric(yy, pp):
    """Helper that cpu metric."""
    return float(np.mean((yy - pp) ** 2))


def _spy_parallel(monkeypatch):
    """Helper that spy parallel."""
    calls = {"n": 0}
    from joblib import Parallel as _RealParallel

    class _CountingParallel:
        """Groups tests covering CountingParallel."""
        def __init__(self, *a, **kw):
            calls["n"] += 1
            self._inner = _RealParallel(*a, **kw)

        def __call__(self, *a, **kw):
            return self._inner(*a, **kw)

    monkeypatch.setattr("joblib.Parallel", _CountingParallel)
    return calls


def test_gpu_bound_metric_fn_forces_serial_and_warns(monkeypatch):
    """Gpu bound metric fn forces serial and warns."""
    calls = _spy_parallel(monkeypatch)
    y, p = _data()
    with pytest.warns(RuntimeWarning, match="GPU-bound"):
        bootstrap_metrics(y, p, {"gpu": _gpu_bound_metric}, n_bootstrap=_R, random_state=7, n_jobs=4)
    assert calls["n"] == 0, "a GPU-bound metric_fns callable must skip the joblib parallel resample loop entirely"


def test_gpu_bound_metric_fn_ignores_backend_env_override(monkeypatch):
    """Gpu bound metric fn ignores backend env override."""
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_BACKEND", "loky")
    calls = _spy_parallel(monkeypatch)
    y, p = _data()
    with pytest.warns(RuntimeWarning, match="GPU-bound"):
        bootstrap_metrics(y, p, {"gpu": _gpu_bound_metric}, n_bootstrap=_R, random_state=7, n_jobs=4)
    assert calls["n"] == 0


def test_cpu_only_metric_fn_still_uses_parallel_pool(monkeypatch):
    """Cpu only metric fn still uses parallel pool."""
    calls = _spy_parallel(monkeypatch)
    y, p = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        bootstrap_metrics(y, p, {"cpu": _cpu_metric}, n_bootstrap=_R, random_state=7, n_jobs=4)
    assert calls["n"] == 1, "a CPU-only metric_fns callable must keep using the parallel resample loop"
