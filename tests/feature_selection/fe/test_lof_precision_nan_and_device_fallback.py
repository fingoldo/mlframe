"""Regression tests: LOF reference distances are formed in double on centred data, non-finite replay rows score NaN, device faults and KTC failures warn."""

import logging
import sys
import types

import numpy as np

from mlframe.feature_selection.filters import _lof_fe, _lof_gpu
from mlframe.utils.log_throttle import reset_throttle_counts


def test_lof_scores_float32_far_from_origin_match_double(monkeypatch):
    """float32 data far from the origin must give the same LOF as the double-precision computation of the same values."""
    rng = np.random.default_rng(0)
    X = (1000.0 + rng.normal(size=(300, 3))).astype(np.float32)
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "1")
    got = _lof_fe.lof_scores(X, k=10)
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "0")
    want = _lof_fe.lof_scores(X.astype(np.float64), k=10)
    np.testing.assert_allclose(got, want, rtol=1e-3)


def test_lof_transform_scores_nonfinite_rows_nan(monkeypatch):
    """A replay row with NaN or inf must score NaN, the finite rows keep their score."""
    monkeypatch.setattr(_lof_gpu, "use_device_lof", lambda n: False)
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(200, 2)).astype(np.float32)
    lrd, kd, k_eff = _lof_fe._lof_fit_reference(ref, 10)
    new = rng.normal(size=(5, 2)).astype(np.float32)
    new[1, 0] = np.nan
    new[3, 1] = np.inf
    got = _lof_fe._lof_transform(new, ref, lrd, kd, k_eff)
    assert np.isnan(got[[1, 3]]).all()
    assert np.isfinite(got[[0, 2, 4]]).all()


def test_lof_gpu_launch_fault_returns_none_and_warns(monkeypatch, caplog):
    """A device fault must hand the rows back to the host scorer with a warning instead of raising."""
    reset_throttle_counts()

    def boom(a):
        """Fail like an out-of-memory allocation."""
        raise RuntimeError("out of memory")

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=boom))
    ref = np.zeros((30, 3), dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger=_lof_gpu.logger.name):
        got = _lof_gpu.lof_transform_gpu(np.zeros((4, 3), dtype=np.float32), ref, np.ones(30, np.float32), np.ones(30, np.float32), 5)
    assert got is None
    assert any("device LOF failed" in r.getMessage() for r in caplog.records)


def test_lof_ktc_choose_failure_warns(monkeypatch, caplog):
    """A failing kernel-tuning lookup must log a warning, not a debug line."""
    reset_throttle_counts()
    monkeypatch.delenv("MLFRAME_LOF_DEVICE", raising=False)

    class _Spec:
        """A spec whose lookup fails."""

        def choose(self, **kw):
            """Always fail."""
            raise RuntimeError("cache unreadable")

    monkeypatch.setattr(_lof_gpu, "_SPEC", _Spec())
    with caplog.at_level(logging.WARNING, logger=_lof_gpu.logger.name):
        _lof_gpu.use_device_lof(10)
    assert any("KTC choose() failed" in r.getMessage() for r in caplog.records)
