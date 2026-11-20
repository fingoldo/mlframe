"""Every silent slow-path fallback in the usability pool / wavelet FE is logged at WARNING (throttled), driven on the CPU by monkeypatching."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest
from tests._cuda_require import require_cuda_device

from mlframe.utils.log_throttle import reset_throttle_counts

F = "mlframe.feature_selection.filters."


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget throttle counts so each test sees its first record."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def _boom(*a, **k):
    """Synthetic device fault."""
    raise RuntimeError("synthetic device fault")


def test_pair_combo_gpu_failure_is_warned(monkeypatch, caplog):
    """A failing cupy pair-combo twin routes to CPU with a WARNING, not a debug line."""
    from mlframe.feature_selection.filters import _usability_njit_pool as P

    monkeypatch.setattr(P, "_CUPY_AVAIL", True)
    monkeypatch.setenv("MLFRAME_USABILITY_POOL_BACKEND", "gpu")
    monkeypatch.setattr(F + "_gpu_policy.gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(P, "_pair_combo_mi_cupy", _boom)
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 3, n).astype(np.int64)
    cy = np.bincount(y) / n
    h_y = float(-(cy * np.log(cy)).sum())
    caplog.set_level(logging.WARNING)
    out = P.score_pair_combos(rng.standard_normal(n), rng.standard_normal(n), y, (y, h_y, 3), 5, [0, 3], [0, 3], [0, 1])
    assert out.shape == (8,)
    assert _warned(caplog, "falling through to CPU")


def test_usability_backend_choice_failure_is_warned(monkeypatch, caplog):
    """A failing tuner lookup falls back to the size heuristic with a WARNING."""
    from mlframe.feature_selection.filters import _usability_njit_pool as P

    monkeypatch.delenv("MLFRAME_USABILITY_POOL_BACKEND", raising=False)
    monkeypatch.setattr(P, "_USABILITY_PARALLELISM_SPEC", types.SimpleNamespace(choose=_boom))
    caplog.set_level(logging.WARNING)
    assert P._usability_backend_choice(1000, 100) == "parallel"
    assert _warned(caplog, "choose() failed")


def test_resident_table_gpu_failure_is_warned(monkeypatch, caplog):
    """A device fault inside the resident pair-combo table returns None with a WARNING."""
    from mlframe.feature_selection.filters import _usability_pool_resident as R

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace())
    monkeypatch.setattr(F + "_gpu_policy.gpu_globally_disabled", lambda: False)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100).astype(np.int64)
    caplog.set_level(logging.WARNING)
    out = R.score_pair_combos_table_resident([(rng.standard_normal(100), rng.standard_normal(100))], y, (y, 0.69, 2), 4, np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    assert out is None
    assert _warned(caplog, "GPU path failed")


def test_resident_chunk_write_failure_is_throttled(monkeypatch, caplog):
    """The per-chunk resident write failure logs a bounded number of WARNINGs however many chunks fail."""
    from mlframe.feature_selection.filters import _usability_pool_resident as R

    fake_cp = types.SimpleNamespace(asnumpy=np.asarray, asarray=np.asarray, ascontiguousarray=np.ascontiguousarray)
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setattr(F + "_gpu_resident_select._radix_select_interior_edges", _boom)
    monkeypatch.setattr(F + "_usability_njit_pool._gpu_quantile_bin_codes", lambda c, q: (np.zeros(c.shape, dtype=np.int64), np.ones(c.shape[0], dtype=np.int64)))
    monkeypatch.setattr(F + "_usability_njit_pool._gpu_marginal_mi", lambda *a: np.zeros(3))
    caplog.set_level(logging.WARNING)
    table = np.zeros(3)
    for _ in range(20):
        R._write_combo_chunk_mi(np.zeros((3, 50)), np.ones(3, dtype=bool), True, 4, 2, 0.5, 2, np.zeros(50, dtype=np.int64), np.linspace(0, 1, 5), 50, table, 0, 3)
    n_warn = sum(1 for r in caplog.records if r.levelno >= logging.WARNING and "resident MI-table write failed" in r.getMessage())
    assert 1 <= n_warn <= 5


def test_row_stat_pool_failure_is_warned(monkeypatch, caplog):
    """A failing row-statistic family drops its candidates with a WARNING."""
    from mlframe.feature_selection.filters._usability_row_stat_pool import row_stat_pool_candidates

    monkeypatch.setattr(F + "_row_stat_fe.hybrid_row_stat_fe", _boom)
    caplog.set_level(logging.WARNING)
    df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)[::-1]})
    assert row_stat_pool_candidates(df, np.arange(50.0), ["a", "b"], np.float32, 8) == []
    assert _warned(caplog, "row-statistic usability candidates skipped")


def test_warp_pool_failure_is_warned(monkeypatch, caplog):
    """A failing warp family drops its candidates with a WARNING."""
    from mlframe.feature_selection.filters._usability_warp_pool import warp_pool_candidates

    monkeypatch.setattr(F + "_oof_warp_fe.warp_candidates", _boom)
    caplog.set_level(logging.WARNING)
    df = pd.DataFrame({"a": np.arange(50.0)})
    assert warp_pool_candidates(df, np.arange(50.0), ["a"], np.float32, 8) == []
    assert _warned(caplog, "warp usability candidates skipped")


def test_pool_resident_ktc_strict_check_failure_is_warned(monkeypatch, caplog):
    """A failing strict-GPU probe is logged before the crossover decision."""
    from mlframe.feature_selection.filters import _usability_pool_resident_ktc as K

    monkeypatch.setattr(F + "_gpu_policy.gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(F + "_fe_gpu_strict.fe_gpu_strict_enabled", _boom)
    monkeypatch.setattr(K, "_POOLRES_SPEC", None)
    caplog.set_level(logging.WARNING)
    assert K.pool_table_use_resident(1000, 4, 96) is False
    assert _warned(caplog, "fe_gpu_strict_enabled() check failed")


def test_wavelet_binned_mi_gpu_failure_is_warned(monkeypatch, caplog):
    """A failing device binned MI falls back to the host MI with a WARNING."""
    from mlframe.feature_selection.filters import _wavelet_basis_fe as W

    monkeypatch.setattr(W, "_binnedmi_gpu_enabled", lambda **k: True)
    monkeypatch.setattr(W, "_binned_mi_cupy", _boom)
    rng = np.random.default_rng(0)
    caplog.set_level(logging.WARNING)
    assert W._binned_mi(rng.standard_normal(300), rng.integers(0, 2, 300)) >= 0.0
    assert _warned(caplog, "_binned_mi_cupy failed")


def test_wavelet_gpu_leg_selection_failure_is_warned(monkeypatch, caplog):
    """A failing device leg selection returns None (host path) with a WARNING."""
    from mlframe.feature_selection.filters import _wavelet_basis_fe as W

    monkeypatch.setattr(F + "_wavelet_basis_fe_batched.select_wavelet_legs_batched", _boom)
    caplog.set_level(logging.WARNING)
    assert W._try_gpu_leg_selection(np.arange(300.0), np.arange(300) % 2, 0.0, 300.0, 3, 6, 3.5, False) is None
    assert _warned(caplog, "GPU-resident leg-selection path failed")


def test_wavelet_haar_choose_failure_is_throttled(monkeypatch, caplog):
    """The per-leg tuner failure logs a bounded number of WARNINGs across many legs."""
    from mlframe.feature_selection.filters import _wavelet_basis_fe as W

    monkeypatch.setattr(W, "_HAAR_LEG_PARALLELISM_SPEC", types.SimpleNamespace(choose=_boom))
    caplog.set_level(logging.WARNING)
    z = np.linspace(0, 1, 50)
    for _ in range(30):
        W._dyadic_haar_leg(z, 1, 0)
    n_warn = sum(1 for r in caplog.records if "choose() failed" in r.getMessage())
    assert 1 <= n_warn <= 5


def test_wavelet_device_born_failure_is_warned(monkeypatch, caplog):
    """A failing device-born leg scoring returns None to the caller with a WARNING."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _wavelet_basis_fe_batched as B

    monkeypatch.setattr(F + "_fe_resident_operands.resident_operand", _boom)
    rng = np.random.default_rng(0)
    x = rng.random(400)
    caplog.set_level(logging.WARNING)
    assert B._select_wavelet_legs_batched_device(x, (x > 0.5).astype(int), 0.0, 1.0, max_scale=3, max_legs=6, scale_sigma=3.5) is None
    assert _warned(caplog, "device batched-MI path failed")


def test_wavelet_resident_selection_failure_is_warned(monkeypatch, caplog):
    """A failing strict-residency probe is logged before the host body runs."""
    require_cuda_device()
    from mlframe.feature_selection.filters import _wavelet_basis_fe_batched as B

    monkeypatch.setattr(F + "_gpu_strict_fe.fe_gpu_device_born_wavelet_enabled", _boom)
    rng = np.random.default_rng(0)
    x = rng.random(400)
    caplog.set_level(logging.WARNING)
    B.select_wavelet_legs_batched(x, (x > 0.5).astype(int), 0.0, 1.0, max_scale=3, max_legs=6, scale_sigma=3.5)
    assert _warned(caplog, "resident wavelet-leg computation failed")


def test_wavelet_legs_tuner_choose_failure_is_warned(monkeypatch, caplog):
    """A failing backend tuner lookup falls back to the row-count rule with a WARNING."""
    from mlframe.feature_selection.filters import _wavelet_legs_tuner as T

    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    monkeypatch.setattr(T, "_SPEC", types.SimpleNamespace(choose=_boom))
    caplog.set_level(logging.WARNING)
    assert T.wavelet_device_chosen(T.DEVICE_FALLBACK_ROWS) is True
    assert _warned(caplog, "KTC choose() failed")


def test_y_encoding_broken_xxhash_is_warned(monkeypatch, caplog):
    """An installed but unusable xxhash is reported, not silently replaced by blake2b."""
    from mlframe.feature_selection.filters import _y_encoding as Y

    fake = types.ModuleType("xxhash")

    def _raise(name):
        """Module attribute access fails like a broken install."""
        raise RuntimeError("broken native lib")

    fake.__getattr__ = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xxhash", fake)
    caplog.set_level(logging.WARNING)
    h = Y._resolve_content_hash()
    assert isinstance(h(memoryview(b"abc")), int)
    assert _warned(caplog, "xxhash is installed but unusable")
