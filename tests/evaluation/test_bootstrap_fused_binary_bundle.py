"""Regression + perf-sentinel tests for the fused auc/brier/log_loss/ece batch bootstrap.

``bootstrap_auc_brier_ll_ece_batch`` fuses the roc_auc/brier/log_loss/ece bundle
``honest_diagnostics._bootstrap_block`` computes together into ONE prange-parallel njit call,
replacing ``bootstrap_metrics``'s per-resample Python-dispatch loop. Must be bit-identical
(modulo the documented ~1e-14 FP-reorder from calling the sequential brier/log_loss kernel
instead of nesting a parallel-reduction kernel inside its own outer prange loop).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.calibration.policy import _ece_score
from mlframe.evaluation._bootstrap_fused_binary_bundle import bootstrap_auc_brier_ll_ece_batch
from mlframe.evaluation.bootstrap import bootstrap_metrics
from mlframe.evaluation._bootstrap_jackknife import _jackknife_auc
from mlframe.metrics.core import (
    fast_brier_score_loss as _fast_brier,
    fast_log_loss as _fast_ll,
    make_bootstrap_auc_resampler,
)

_TOL = 1e-9


def _make_binary_fixture(n: int, seed: int = 1):
    """Continuous, tie-free probability scores (no clip -- clipping collapses values onto a
    shared boundary and trips the tie-free gate)."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.3).astype(np.float64)
    raw = y_true * 1.5 + rng.normal(size=n) * 0.5
    p_pos = 1.0 / (1.0 + np.exp(-raw))
    return y_true, p_pos


def _reference_bootstrap_metrics(y_true, p_pos, n_bootstrap, random_state):
    """Replicates honest_diagnostics._bootstrap_block's exact metric_fns/jackknife_fns wiring."""

    def _brier(yy, pp):
        """Bootstrap-resample Brier score via the numba-fast kernel."""
        return float(_fast_brier(yy, pp))

    def _ll(yy, pp):
        """Bootstrap-resample log-loss via the numba-fast kernel."""
        return float(_fast_ll(yy, pp))

    def _ll_per_row(yy, pp):
        """Per-row cross-entropy contribution for the BCa jackknife's O(n) exact leave-one-out formula."""
        eps = np.finfo(np.asarray(pp).dtype).eps
        pc = np.clip(pp, eps, 1.0 - eps)
        return np.where(np.asarray(yy) == 1, -np.log(pc), -np.log(1.0 - pc))

    def _brier_per_row(yy, pp):
        """Per-row squared-error contribution for the BCa jackknife's O(n) exact leave-one-out formula."""
        d = np.asarray(pp, dtype=np.float64) - np.asarray(yy, dtype=np.float64)
        return d * d

    metric_fns = {"brier": _brier, "log_loss": _ll, "ece": lambda yy, pp: _ece_score(yy, pp)}
    per_row_fns = {"log_loss": (_ll_per_row, True, None), "brier": (_brier_per_row, False, None)}
    jackknife_fns = {"roc_auc": lambda yy, ss: _jackknife_auc(yy, ss)}
    metric_fns_idx = {"roc_auc": make_bootstrap_auc_resampler(y_true, p_pos)}
    return bootstrap_metrics(
        y_true, p_pos, metric_fns, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y_true,
        random_state=random_state, metric_fns_idx=metric_fns_idx, per_row_fns=per_row_fns, jackknife_fns=jackknife_fns,
    )


@pytest.mark.parametrize("n,n_bootstrap", [(5_000, 100), (50_000, 200)])
def test_fused_bundle_matches_bootstrap_metrics(n, n_bootstrap):
    """All four metrics' point/lo/hi/samples must match bootstrap_metrics to within the documented
    ~1e-14 FP-reorder tolerance (sequential vs size-adaptive parallel-reduction kernel choice)."""
    y_true, p_pos = _make_binary_fixture(n)
    ref = _reference_bootstrap_metrics(y_true, p_pos, n_bootstrap, random_state=0)
    fused = bootstrap_auc_brier_ll_ece_batch(y_true, p_pos, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y_true, random_state=0)
    assert fused is not None
    for name in ("roc_auc", "brier", "log_loss", "ece"):
        r, f = ref[name], fused[name]
        assert abs(r["point"] - f["point"]) < _TOL, f"{name} point mismatch"
        assert abs(r["lo"] - f["lo"]) < _TOL, f"{name} lo mismatch"
        assert abs(r["hi"] - f["hi"]) < _TOL, f"{name} hi mismatch"
        assert np.allclose(np.sort(r["samples"]), np.sort(f["samples"]), atol=_TOL), f"{name} samples mismatch"


@pytest.mark.parametrize("n,n_bootstrap", [(2_000, 100), (20_000, 100)])
def test_fused_bundle_matches_bootstrap_metrics_on_tied_scores(n, n_bootstrap):
    """Tied/low-cardinality base scores route through the grouped (tie-aware) batch kernel instead of
    falling back to bootstrap_metrics -- must still match it to the documented tolerance. Regression
    guard for the fused bundle's former hard ``tie_free`` gate, which used to return None (forcing the
    fully serial fallback) on ANY duplicate score -- the common case for real quantised model outputs."""
    rng = np.random.default_rng(3)
    p_pos = np.round(rng.random(n) * 9) / 9.0  # low-cardinality -> guaranteed ties
    y_true = (rng.random(n) < 0.3).astype(np.float64)
    ref = _reference_bootstrap_metrics(y_true, p_pos, n_bootstrap, random_state=0)
    fused = bootstrap_auc_brier_ll_ece_batch(y_true, p_pos, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y_true, random_state=0)
    assert fused is not None
    for name in ("roc_auc", "brier", "log_loss", "ece"):
        r, f = ref[name], fused[name]
        assert abs(r["point"] - f["point"]) < _TOL, f"{name} point mismatch"
        assert abs(r["lo"] - f["lo"]) < _TOL, f"{name} lo mismatch"
        assert abs(r["hi"] - f["hi"]) < _TOL, f"{name} hi mismatch"
        assert np.allclose(np.sort(r["samples"]), np.sort(f["samples"]), atol=_TOL), f"{name} samples mismatch"


def test_fused_bundle_handles_all_tied_scores():
    """Degenerate case: every base score identical (ngroups=1). Must not crash and must match
    bootstrap_metrics (AUC is nan throughout since there is no ranking signal)."""
    rng = np.random.default_rng(4)
    n = 500
    p_pos = np.full(n, 0.5)
    y_true = (rng.random(n) < 0.3).astype(np.float64)
    ref = _reference_bootstrap_metrics(y_true, p_pos, 50, random_state=0)
    fused = bootstrap_auc_brier_ll_ece_batch(y_true, p_pos, n_bootstrap=50, alpha=0.05, stratify=y_true, random_state=0)
    assert fused is not None
    for name in ("brier", "log_loss", "ece"):
        assert abs(ref[name]["point"] - fused[name]["point"]) < _TOL, f"{name} point mismatch"


def test_fused_bundle_faster_than_bootstrap_metrics():
    """Perf sentinel: measured 2.9x-3.4x@50k-300k; assert >=1.5x to catch a regression without
    flaking on a loaded runner."""
    y_true, p_pos = _make_binary_fixture(30_000, seed=5)
    n_bootstrap = 150

    # warm both paths' JIT compilation before timing
    _reference_bootstrap_metrics(y_true[:1000], p_pos[:1000], 4, random_state=999)
    bootstrap_auc_brier_ll_ece_batch(y_true[:1000], p_pos[:1000], n_bootstrap=4, stratify=y_true[:1000], random_state=999, chunk_size=4)

    t0 = time.perf_counter()
    _reference_bootstrap_metrics(y_true, p_pos, n_bootstrap, random_state=1)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    bootstrap_auc_brier_ll_ece_batch(y_true, p_pos, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y_true, random_state=1)
    t_fused = time.perf_counter() - t0

    speedup = t_ref / t_fused
    assert speedup >= 1.5, f"fused bundle not faster: {speedup:.2f}x (ref={t_ref * 1e3:.1f}ms fused={t_fused * 1e3:.1f}ms)"
