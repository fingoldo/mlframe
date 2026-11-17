"""Regression tests: ksg_lnc_mi fallback forwards seed/intens, LNC jitter survives a large offset, GPU mixed-KSG jitter matches CPU, dispatcher logs a missing cupy."""

import logging
import sys
import types

import numpy as np

from mlframe.feature_selection.filters import _ksg
from mlframe.utils.log_throttle import reset_throttle_counts


def test_lnc_low_entropy_fallback_forwards_seed_and_intens(monkeypatch):
    """The low-entropy fallback to mixed-KSG must pass the caller's seed and intens through."""
    seen = {}

    def fake(x, y, k=5, use_gpu=False, intens=1e-10, max_input_n=50000, seed=0):
        """Record the forwarded arguments."""
        seen.update(intens=intens, seed=seed)
        return 0.0

    monkeypatch.setattr(_ksg, "mixed_ksg_mi", fake)
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = (rng.random(500) > 0.5).astype(float)
    _ksg.ksg_lnc_mi(x, y, intens=1e-7, seed=11)
    assert seen == {"intens": 1e-7, "seed": 11}


def test_lnc_tie_jitter_survives_large_offset():
    """A 1e8 offset must not change the LNC estimate of tied integer data (the jitter is applied after centring)."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 40, 400).astype(np.float64)
    y = x + rng.integers(0, 3, 400)
    base = _ksg.ksg_lnc_mi(x, y, low_entropy_skip=False)
    shifted = _ksg.ksg_lnc_mi(x + 1e8, y + 1e8, low_entropy_skip=False)
    assert abs(base - shifted) < 1e-6


def _fake_cupy():
    """A numpy-backed stand-in exposing the cupy calls mixed_ksg_mi_gpu makes."""

    class _A(np.ndarray):
        """ndarray with the ``get`` of a device array."""

        def get(self):
            """Host copy."""
            return np.asarray(self)

    ns = types.SimpleNamespace(int64=np.int64)
    ns.asarray = lambda a: np.asarray(a).view(_A)
    ns.searchsorted = lambda a, v, side="left": np.searchsorted(np.asarray(a), np.asarray(v), side=side).view(_A)
    ns.maximum = lambda a, b: np.maximum(np.asarray(a), b).view(_A)
    return ns


def test_gpu_mixed_ksg_jitter_matches_cpu_on_large_offset(monkeypatch):
    """The device path must break ties relative to the column spread, as the CPU path does."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(_ksg, "device_sort", np.sort)
    rng = np.random.default_rng(2)
    x = 1e8 + rng.integers(0, 3, 600).astype(np.float64)
    cpu = _ksg.mixed_ksg_mi(x, x)
    gpu = _ksg.mixed_ksg_mi_gpu(x, x)
    assert abs(cpu - np.log(3)) < 0.25
    assert abs(gpu - cpu) < 1e-6


def test_dispatch_logs_missing_cupy(monkeypatch, caplog):
    """A large-n dispatch on a host without cupy must warn that it runs on the CPU."""
    reset_throttle_counts()
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setenv("MLFRAME_KSG_GPU_N", "100")
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    with caplog.at_level(logging.WARNING, logger=_ksg.logger.name):
        _ksg.ksg_mi_dispatch(x, x + rng.normal(size=300))
    assert any("cupy unavailable" in r.getMessage() for r in caplog.records)
