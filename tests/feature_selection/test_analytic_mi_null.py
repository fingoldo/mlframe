"""Analytic large-n MI null (2026-06-16): equivalence to the permutation null + the speed win.

At n >= threshold, mi_direct(return_null_mean=...) replaces the permutation null with the analytic
Miller-Madow null mean + G-test p-value (see _analytic_mi_null). These tests pin: (1) the closed-form
matches the permutation kernel's null mean and is decision-equivalent on the p-value, (2) the off-switch
restores the permutation path, (3) below threshold the analytic path stays dormant (byte-identical),
(4) the analytic path is dramatically faster (the biz-value win that motivated it).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection.filters.permutation import mi_direct
from mlframe.feature_selection.filters._analytic_mi_null import (
    analytic_mi_null,
    analytic_null_min_n,
)

NB = 10


def _binned(n, signal, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, NB, n).astype(np.int32)
    if signal > 0:
        y = np.where(rng.random(n) < signal, x, rng.integers(0, NB, n)).astype(np.int32)
    else:
        y = rng.integers(0, NB, n).astype(np.int32)
    return np.column_stack([x, y]).astype(np.int32), np.array([NB, NB], dtype=np.int32)


def _md(data, nbins, **kw):
    return mi_direct(data, x=(0,), y=(1,), factors_nbins=nbins, npermutations=64,
                     min_nonzero_confidence=0.0, parallelism="none", prefer_gpu=False,
                     return_null_mean=True, **kw)


def test_analytic_formula_miller_madow():
    # null_mean = (Bx-1)(By-1)/(2N); p in [0,1]; df<=0 / mi<=0 -> non-significant.
    nm, p = analytic_mi_null(0.01, 100_000, 10, 10)
    assert nm == pytest.approx((9 * 9) / (2 * 100_000))
    assert 0.0 <= p <= 1.0
    assert analytic_mi_null(0.0, 100_000, 10, 10) == (pytest.approx(81 / 200_000), 1.0)
    assert analytic_mi_null(0.5, 1000, 1, 5) == (0.0, 1.0)  # df<=0 degenerate


@pytest.mark.parametrize("signal", [0.0, 0.05])
def test_analytic_matches_permutation_large_n(monkeypatch, signal):
    n = max(int(analytic_null_min_n()), 60_000)
    data, nbins = _binned(n, signal, seed=11)

    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
    mi_a, conf_a, nm_a, p_a = _md(data, nbins)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    mi_p, conf_p, nm_p, p_p = _md(data, nbins)

    # observed MI is computed the same way on both paths -> identical.
    assert mi_a == pytest.approx(mi_p, rel=1e-9)
    # analytic null mean matches the permutation null mean closely.
    assert nm_a == pytest.approx(nm_p, rel=0.20, abs=1e-4)
    # decision-equivalence at the canonical alpha=0.05: both agree significant / not.
    assert (p_a < 0.05) == (p_p < 0.05)
    if signal > 0:
        assert p_a < 0.05 and p_p < 0.05  # genuine signal -> significant on both


def test_below_threshold_is_dormant(monkeypatch):
    # n < threshold: toggling the analytic flag must NOT change the result (permutation path both).
    n = max(1, int(analytic_null_min_n()) // 10)
    data, nbins = _binned(n, 0.05, seed=7)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
    r_on = _md(data, nbins)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    r_off = _md(data, nbins)
    assert r_on == pytest.approx(r_off, rel=1e-9, abs=1e-12)


def test_analytic_is_faster_large_n(monkeypatch):
    n = max(int(analytic_null_min_n()), 120_000)
    data, nbins = _binned(n, 0.05, seed=5)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
    _md(data, nbins)  # warmup
    t = time.perf_counter()
    for _ in range(10):
        _md(data, nbins)
    t_a = time.perf_counter() - t
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    _md(data, nbins)
    t = time.perf_counter()
    for _ in range(10):
        _md(data, nbins)
    t_p = time.perf_counter() - t
    # analytic skips the permutation shuffles entirely -> materially faster (conservative bar: 3x).
    assert t_a < t_p / 3.0, f"analytic not faster: analytic={t_a:.3f}s permutation={t_p:.3f}s"
