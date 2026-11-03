"""The per-column honest null on the edge pool gives the serial loop's exceedance count (permutations seeded per index)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _honest_mdlp_null as H


def _hook(x: np.ndarray, y_raw: np.ndarray):
    """A honest-codes hook whose single-column binning depends on the permuted target: cut at the mean of x over the permuted positives."""

    def codes_fn(row_perm):
        """Whole-frame fallback (not used when the column hook answers)."""
        raise AssertionError("the column hook should answer")

    def column(row_perm, col):
        """Codes of the column under the permuted target."""
        yp = y_raw[np.asarray(row_perm)]
        cut = float(x[yp == 1].mean()) if (yp == 1).any() else 0.0
        return (x > cut).astype(np.int64) + (x > cut + 1.0).astype(np.int64), 3

    codes_fn.column = column
    return codes_fn


def test_pool_equals_serial(monkeypatch):
    """Same exceedance count and observed MI for a signal column, pool on and off."""
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.standard_normal(n)
    y = (x + 1.5 * rng.standard_normal(n) > 0).astype(np.int64)
    codes = (x > 0).astype(np.int64) + (x > 1.0).astype(np.int64)
    factors = np.column_stack([codes, y]).astype(np.int64)
    hook = _hook(x, y)
    monkeypatch.setenv("MLFRAME_HONEST_COLUMN_POOL", "0")
    want = H.honest_candidate_nfailed(hook, factors, [3, 2], 1, 0, 40, seed=7)
    monkeypatch.setenv("MLFRAME_HONEST_COLUMN_POOL", "1")
    called = []
    real = H._column_nfailed_pooled
    monkeypatch.setattr(H, "_column_nfailed_pooled", lambda *a, **k: (lambda r: called.append(r is not None) or r)(real(*a, **k)))
    got = H.honest_candidate_nfailed(hook, factors, [3, 2], 1, 0, 40, seed=7)
    assert called == [True]
    assert got == want and want is not None


def test_fdr_floor_pool_equals_serial(monkeypatch):
    """The honest FDR floor with the permutations on the edge pool equals the serial floor (whole-frame re-binning hook, target-dependent codes)."""
    rng = np.random.default_rng(1)
    n = 3000
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(np.int64)

    def codes_fn(row_perm):
        """Every column cut at the mean over the permuted positives (target-dependent codes for the whole frame)."""
        yp = y[np.asarray(row_perm)]
        cuts = X[yp == 1].mean(axis=0)
        return (X > cuts).astype(np.int64), np.full(4, 2)

    factors = np.column_stack([(X > X[y == 1].mean(axis=0)).astype(np.int64), y]).astype(np.int64)
    nb = np.array([2, 2, 2, 2, 2])
    monkeypatch.setenv("MLFRAME_HONEST_FDR_POOL", "0")
    want = H.honest_fdr_floor(codes_fn, factors, nb, 4, [0, 1, 2, 3], 30, 0.95, seed=3)
    monkeypatch.setenv("MLFRAME_HONEST_FDR_POOL", "1")
    got = H.honest_fdr_floor(codes_fn, factors, nb, 4, [0, 1, 2, 3], 30, 0.95, seed=3)
    assert got == want
