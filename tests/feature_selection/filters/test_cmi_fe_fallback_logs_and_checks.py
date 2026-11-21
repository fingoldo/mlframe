"""Regression tests: silent GPU/KTC fallbacks in the CMI-greedy and meta-fingerprint modules log at WARNING; row-count mismatches are rejected."""
import logging
import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


def _warned(caplog, needle):
    """True when a WARNING record contains ``needle``."""
    return any(r.levelno >= logging.WARNING and needle in r.getMessage() for r in caplog.records)


def test_meta_fingerprint_device_fallback_warns(monkeypatch, caplog):
    """A missing cupy sends the block to the host and says so at WARNING."""
    from mlframe.feature_selection.filters import _meta_fingerprint_gpu as m

    reset_throttle_counts()
    monkeypatch.setitem(sys.modules, "cupy", None)
    with caplog.at_level(logging.WARNING):
        assert m._moments_device(np.zeros((2, 5)), None) is None
    assert _warned(caplog, "device form is unavailable")


def test_meta_fingerprint_ktc_choose_failure_warns(monkeypatch, caplog):
    """A failing KTC choose() falls back to the row mark and says so at WARNING."""
    from mlframe.feature_selection.filters import _meta_fingerprint_gpu as m

    reset_throttle_counts()
    monkeypatch.delenv("MLFRAME_META_FINGERPRINT_DEVICE", raising=False)

    class _Bad:
        """Spec whose choose() always fails."""

        def choose(self, **kw):
            """Raise."""
            raise RuntimeError("boom")

    monkeypatch.setattr(m, "_SPEC", _Bad())
    with caplog.at_level(logging.WARNING):
        assert m.fingerprint_moments_on_device(10) is False
    assert _warned(caplog, "KTC choose() failed")


def test_column_moments_rejects_short_target():
    """A centred target shorter than the columns would be read out of bounds by the kernels."""
    from mlframe.feature_selection.filters._meta_fingerprint_gpu import column_moments

    with pytest.raises(ValueError):
        column_moments(np.random.default_rng(0).normal(size=(3, 50)), np.zeros(10))


def test_cmi_ktc_choose_failure_warns(monkeypatch, caplog):
    """A failing KTC choose() for the CMI batch backend falls back to the floors and says so at WARNING."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_ktc as k

    reset_throttle_counts()

    class _Bad:
        """Spec whose choose() always fails."""

        def choose(self, **kw):
            """Raise."""
            raise RuntimeError("boom")

    monkeypatch.setattr(k, "_SPEC", _Bad())
    with caplog.at_level(logging.WARNING):
        assert k.candidate_batch_device_chosen(10, 2) is False
    assert _warned(caplog, "KTC choose() failed")


def test_cmi_sweep_host_concurrent_calls_restore_env(monkeypatch):
    """Two overlapping host sweeps must leave the CMI env switches as they found them."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_ktc as k
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_steps as steps

    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    inner_done = threading.Event()
    state = {"depth": 0}

    def _run_inner(y, z):
        """Run a nested sweep, swallowing the injected stop."""
        try:
            k._sweep_host(X, y, z)
        except RuntimeError:
            pass

    def _slow_terms(y, z):
        """First entry spawns a second sweep thread and waits briefly so the two overlap."""
        if state["depth"] == 0:
            state["depth"] = 1
            t = threading.Thread(target=lambda: (_run_inner(y, z), inner_done.set()))
            t.start()
            inner_done.wait(0.5)
            state["thread"] = t
        raise RuntimeError("stop")

    monkeypatch.setattr(steps, "precompute_cmi_yz_terms", _slow_terms)
    X = np.zeros((4, 1), dtype=np.int64)
    with pytest.raises(RuntimeError):
        k._sweep_host(X, X[:, 0], X[:, 0])
    state["thread"].join(5)
    import os

    leaked = [name for name in ("MLFRAME_CMI_GPU", "MLFRAME_FE_GPU_STRICT") if name in os.environ]
    assert not leaked, f"env flags left set after the sweep: {leaked}"


def test_qbin_gpu_failure_warns(monkeypatch, caplog):
    """A failing device quantile binner returns None for the host path and says so at WARNING."""
    from mlframe.feature_selection.filters import _fe_resident_operands as ro
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_binning as b

    reset_throttle_counts()

    def _boom(*a, **k):
        """Raise."""
        raise RuntimeError("no device")

    monkeypatch.setattr(ro, "resident_qbin_codes_host", _boom)
    monkeypatch.setattr(ro, "resident_qbin_codes", _boom)
    with caplog.at_level(logging.WARNING):
        assert b._quantile_bin_gpu(np.arange(10.0), 4) is None
        assert b._quantile_bin_gpu_resident(np.arange(10.0), 4) is None
    assert _warned(caplog, "GPU _quantile_bin failed")
    assert _warned(caplog, "GPU-resident _quantile_bin failed")


def test_ent_from_counts_fallback_warns(monkeypatch, caplog):
    """A failing fused entropy kernel falls back to the reduction kernels and says so at WARNING."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_gpu as g

    reset_throttle_counts()

    class _RK:
        """Stand-in reduction kernel."""

        def __init__(self, *a):
            """Accept the cupy ctor args."""

        def __call__(self, *a):
            """Return a fixed value."""
            return 1.0

    def _zeros(*a, **k):
        """Fail like a broken device."""
        raise RuntimeError("no device")

    fake = types.SimpleNamespace(zeros=_zeros, float64=np.float64, ReductionKernel=_RK)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(g, "_ENT_RK", None)
    monkeypatch.setattr(g, "_NNZ_RK", None)
    with caplog.at_level(logging.WARNING):
        g._ent_from_counts(np.array([1, 2]), 0.5)
    assert _warned(caplog, "fused entropy-nnz kernel failed")


def test_greedy_cmi_rejects_row_mismatch():
    """y shorter than X is rejected before any compiled kernel reads past it."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import greedy_cmi_fe_construct, score_candidates_by_cmi

    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)[::-1]})
    with pytest.raises(ValueError):
        greedy_cmi_fe_construct(X, np.zeros(5, dtype=int))
    with pytest.raises(ValueError):
        score_candidates_by_cmi(X, np.zeros(5, dtype=int))
