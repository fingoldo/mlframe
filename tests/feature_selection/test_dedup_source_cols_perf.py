"""Regression + perf sensors for the row-capped, vectorized ``_dedup_collinear_source_cols``.

The legacy partial-NaN path was a per-pair ``np.corrcoef`` loop -- O(P^2 * N) -- run on the FULL train frame. On a
5M-row well-log frame with many NaN-bearing columns that was the 1h+ hang. The fix caps the correlation rows and
vectorizes the partial-NaN block via ``_pairwise_complete_abs_corr``. These tests pin: (1) exact duplicates are still
dropped after row-capping, (2) the vectorized partial-NaN verdict matches the legacy per-pair path, (3) the pass stays
fast and bounded in N.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup import (
    _dedup_collinear_source_cols,
    _pairwise_complete_abs_corr,
)


def _legacy_partial_pair_corr(a: np.ndarray, b: np.ndarray) -> float:
    """The exact per-pair masked corr the legacy loop computed, for the equivalence check."""
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 8:
        return np.nan
    aa, bb = a[finite], b[finite]
    if aa.std() <= 1e-12 or bb.std() <= 1e-12:
        return np.nan
    return abs(float(np.corrcoef(aa, bb)[0, 1]))


def test_pairwise_complete_matches_per_pair_on_partial_nan():
    """The vectorized pairwise-complete |corr| must equal the legacy per-pair masked np.corrcoef (up to FP noise)."""
    rng = np.random.default_rng(3)
    n, P = 5000, 12
    M = rng.normal(size=(P, n))
    M[rng.random((P, n)) < 0.3] = np.nan  # 30% missing, independent masks
    got = _pairwise_complete_abs_corr(M, M)
    for i in range(P):
        for j in range(P):
            ref = _legacy_partial_pair_corr(M[i], M[j])
            if np.isnan(ref):
                assert not np.isfinite(got[i, j]), f"({i},{j}) should be nan"
            else:
                assert abs(got[i, j] - ref) < 1e-9, f"({i},{j}) {got[i, j]} vs {ref}"


def test_exact_duplicate_dropped_above_row_cap():
    """Row-capping must not miss an exact duplicate on a frame far larger than the cap."""
    import os

    rng = np.random.default_rng(4)
    n = 300_000  # > default 100k cap
    a = rng.normal(size=n)
    a[rng.random(n) < 0.2] = np.nan  # partial-NaN so the vectorized partial path is exercised
    X = pd.DataFrame({"a": a, "a_dup": a.copy(), "b": rng.normal(size=n)})
    assert "MLFRAME_FE_DEDUP_MAX_CORR_ROWS" not in os.environ  # trusting the 100k default here
    kept = _dedup_collinear_source_cols(X, list(X.columns), corr_threshold=0.999)
    assert "a_dup" not in kept, f"exact duplicate survived row-cap: {kept}"
    assert "a" in kept and "b" in kept


def test_row_cap_env_override(monkeypatch):
    """The MLFRAME_FE_DEDUP_MAX_CORR_ROWS override is honored (re-import to re-read the module constant)."""
    monkeypatch.setenv("MLFRAME_FE_DEDUP_MAX_CORR_ROWS", "5000")
    import importlib

    import mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup as mod

    _orig_dict = dict(mod.__dict__)
    mod = importlib.reload(mod)
    try:
        assert mod._MAX_CORR_ROWS == 5000
    finally:
        mod.__dict__.clear()
        mod.__dict__.update(_orig_dict)  # restore default constant for other tests


@pytest.mark.parametrize("N,P", [(2_000_000, 40)])
def test_partial_nan_dedup_is_fast_and_bounded(N, P):
    """The partial-NaN dedup on a large frame must finish quickly (pre-fix this was ~70s+ at N=2M P=40; the row-cap
    makes it near-constant in N). Budget is generous to avoid CI flakiness while still catching an O(N) regression."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=N)
    cols = {}
    for j in range(P):
        v = base * rng.uniform(0.5, 1.5) + rng.normal(size=N) * 0.3
        v[rng.random(N) < 0.25] = np.nan
        cols[f"c{j}"] = v
    X = pd.DataFrame(cols)
    t = time.perf_counter()
    _dedup_collinear_source_cols(X, list(X.columns), corr_threshold=0.999)
    dt = time.perf_counter() - t
    assert dt < 8.0, f"dedup took {dt:.1f}s at N={N} P={P}; row-cap/vectorization regressed to O(P^2*N)"
