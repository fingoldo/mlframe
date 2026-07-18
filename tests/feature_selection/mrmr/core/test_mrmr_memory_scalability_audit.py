"""Regression tests for audits/mrmr_audit_2026-07-16/07_memory_scalability.md.

Finding #1: cluster_stability_selection's pre-clustering correlation step used a hardcoded dense
float64 (n, p) copy and had no analogue of the main fit path's sis_screen_threshold p-cap.
Finding #2: each stability-selection bootstrap replicate stored itself in the shared process-wide
MRMR._FIT_CACHE despite being a guaranteed future cache miss (a different row-subsample every call).
Finding #3 (confirmed not a bug): the per-target multioutput loop now logs progress.
Finding #4: mi_direct's GPU fastpath had no proactive VRAM-headroom check before launch.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._stability_cluster import cluster_stability_selection
from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR


def _selector(Xs, ys):
    Xs = np.asarray(Xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    corr = np.abs([np.corrcoef(Xs[:, i], ys)[0, 1] for i in range(Xs.shape[1])])
    corr = np.nan_to_num(corr)
    return np.argsort(-corr)[:3]


def test_cluster_stability_dtype_is_relaxed_by_default(monkeypatch):
    """finding #1: the correlation-clustering step uses float32 by default (MLFRAME_CRIT_DTYPE_RELAXED),
    not an unconditional float64 dense copy."""
    monkeypatch.delenv("MLFRAME_CRIT_DTYPE_RELAXED", raising=False)
    from mlframe.feature_selection.filters._stability_cluster import _stability_corr_dtype

    assert _stability_corr_dtype() is np.float32
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "0")
    assert _stability_corr_dtype() is np.float64


def test_cluster_stability_selection_p_cap_still_recovers_signal(monkeypatch):
    """finding #1: with a p-cap active (well below the fixture's column count), the informative column
    is still clustered and selected -- the cap bounds clustering COST, not selectability."""
    monkeypatch.setenv("MLFRAME_STABILITY_CLUSTER_MAX_FEATURES", "20")
    rng = np.random.default_rng(0)
    n, p = 300, 80
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 2.0 + rng.standard_normal(n) * 0.1
    chosen, freq, info = cluster_stability_selection(X, y, _selector, n_bootstrap=5, corr_threshold=0.8, pi_threshold=0.4)
    assert 0 in chosen
    assert info["n_clusters"] == p  # p-cap doesn't merge dropped columns; each stays its own singleton


def test_cluster_stability_selection_dataframe_no_python_loop_needed(monkeypatch):
    """finding #1: a homogeneous-numeric DataFrame takes the vectorised whole-frame cast, not the
    per-column python loop -- verified indirectly via correctness (same clustering result as an
    equivalent ndarray input)."""
    rng = np.random.default_rng(1)
    n, p = 200, 10
    z = rng.standard_normal(n)
    X = np.column_stack([z + 0.05 * rng.standard_normal(n) for _ in range(3)] + [rng.standard_normal(n) for _ in range(p - 3)])
    y = z * 2.0 + rng.standard_normal(n) * 0.1
    Xdf = pd.DataFrame(X, columns=[f"c{i}" for i in range(p)])
    chosen_arr, _, info_arr = cluster_stability_selection(X, y, _selector, n_bootstrap=5, corr_threshold=0.8)
    chosen_df, _, info_df = cluster_stability_selection(Xdf, y, _selector, n_bootstrap=5, corr_threshold=0.8)
    assert info_arr["n_clusters"] == info_df["n_clusters"]
    assert sorted(chosen_arr.tolist()) == sorted(chosen_df.tolist())


def test_stability_bootstrap_replicates_skip_fit_cache():
    """finding #2: bootstrap-replicate sub-fits set _skip_fit_cache=True and never appear in the shared
    process-wide MRMR._FIT_CACHE, so they cannot evict an unrelated concurrent caller's cached entry."""
    MRMR._FIT_CACHE.clear()
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    y = pd.Series((X["a"] * 2 > 0).astype(int))
    m = MRMR(
        verbose=0, stability_selection_method="cluster", stability_n_bootstrap=3,
        min_features_fallback=1, fit_cache_max=4,
    )
    m.fit(X, y)
    assert len(MRMR._FIT_CACHE) == 0


def test_multioutput_logs_per_target_progress(caplog):
    """finding #3 (confirmed not a bug, user requested clearer logging): a verbose multioutput fit logs
    which target is currently running."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = pd.DataFrame({"t1": (X["a"] > 0).astype(int), "t2": (X["b"] > 0).astype(int)})
    m = MRMR(verbose=1, multioutput_strategy="union", min_features_fallback=1)
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters.mrmr"):
        m.fit(X, y)
    msgs = [r.message for r in caplog.records]
    assert any("fitting target" in msg and "t1" in msg for msg in msgs)
    assert any("fitting target" in msg and "t2" in msg for msg in msgs)
    assert any("selected" in msg for msg in msgs)


def test_multioutput_subfits_skip_fit_cache():
    """finding #2 (same reasoning applied to multioutput's per-target sub-fits): each target's y_col
    differs, so caching a sub-fit's entry is a guaranteed future miss."""
    MRMR._FIT_CACHE.clear()
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = pd.DataFrame({"t1": (X["a"] > 0).astype(int), "t2": (X["b"] > 0).astype(int)})
    m = MRMR(verbose=0, multioutput_strategy="union", min_features_fallback=1, fit_cache_max=4)
    m.fit(X, y)
    assert len(MRMR._FIT_CACHE) == 0


def test_mi_direct_gpu_dispatch_declines_on_low_vram_cushion(monkeypatch):
    """finding #4: mi_direct's GPU fastpath now proactively declines (falls back to CPU) when
    fe_gpu_has_vram_cushion reports insufficient headroom, instead of only catching the fault
    reactively after an actual launch attempt."""
    import mlframe.feature_selection.filters.permutation as perm_mod

    monkeypatch.setattr(perm_mod, "_MI_DIRECT_GPU_FAILED", False)
    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", lambda: True)
    monkeypatch.setattr(
        "mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda *a, **kw: False
    )
    called = {"gpu": False}

    def _fake_mi_direct_gpu(*args, **kwargs):
        called["gpu"] = True
        raise AssertionError("mi_direct_gpu must not be called when the VRAM cushion check declines")

    monkeypatch.setattr("mlframe.feature_selection.filters.gpu.mi_direct_gpu", _fake_mi_direct_gpu)

    rng = np.random.default_rng(0)
    n = 500
    factors_data = rng.integers(0, 5, size=(n, 3)).astype(np.int32)
    factors_nbins = np.full(3, 5, dtype=np.int32)
    x = (0,)
    y = (1,)
    perm_mod.mi_direct(
        factors_data=factors_data, x=x, y=y, factors_nbins=factors_nbins,
        npermutations=64, prefer_gpu=True, parallelism="none",
    )
    assert called["gpu"] is False
