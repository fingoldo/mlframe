"""Regression tests for the adaptive-Fourier/chirp detector column cap (2026-07-09 fix).

cProfile evidence (bench_mrmr_pre_categorize_family_profile.py, p=420/n=60000): the adaptive/chirp
frequency DETECTOR (``_detect_fourier_freqs_for_col``) is the single most expensive default-ON pre-FE
family, ~34% of the whole pre-categorize wall -- roughly linear in column count regardless of row
count, unlike the rest of the pipeline (already row-subsampled). ``max_adaptive_cols`` bounds how many
columns run the expensive detector; columns beyond the cap still get the cheap fixed-grid Fourier basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import (
    generate_extra_basis_features,
)


def _make_frame(n=2000, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(p)})
    y = np.sin(2.3 * X["f0"].to_numpy()) + 0.1 * rng.standard_normal(n)  # oscillatory in f0
    return X, y


def test_max_adaptive_cols_none_preserves_legacy_behavior():
    X, y = _make_frame(p=6)
    eng_unbounded, meta_unbounded = generate_extra_basis_features(
        X,
        extra_bases=("fourier",),
        y=y,
        fourier_adaptive=True,
        max_adaptive_cols=None,
    )
    eng_explicit_none, meta_explicit_none = generate_extra_basis_features(
        X,
        extra_bases=("fourier",),
        y=y,
        fourier_adaptive=True,
    )  # omitted param -> same default
    assert set(eng_unbounded.columns) == set(eng_explicit_none.columns)


def test_max_adaptive_cols_zero_disables_adaptive_detection_but_keeps_fixed_grid():
    X, y = _make_frame(p=6)
    eng_capped, meta_capped = generate_extra_basis_features(
        X,
        extra_bases=("fourier",),
        fourier_freqs=(1.0, 2.0),
        y=y,
        fourier_adaptive=True,
        max_adaptive_cols=0,
    )
    # Fixed-grid legs (freq 1.0, 2.0) must still be emitted for every column.
    fixed_grid_cols = [c for c in eng_capped.columns if meta_capped[c].get("adaptive") is False]
    assert len(fixed_grid_cols) > 0
    # No adaptive-tagged column may exist when the cap is 0.
    adaptive_cols = [c for c in eng_capped.columns if meta_capped[c].get("adaptive") is True]
    assert adaptive_cols == []


def test_max_adaptive_cols_bounds_which_columns_get_adaptive_detection():
    X, y = _make_frame(p=8, seed=3)
    # All 8 columns are independent standard-normal noise except f0 (oscillatory) -- the adaptive
    # detector should be able to find f0's frequency when unbounded, but with a cap of 1 (only column
    # index 0 = "f0" is eligible), later columns must never get an adaptive-tagged column.
    eng, meta = generate_extra_basis_features(
        X,
        extra_bases=("fourier",),
        cols=list(X.columns),
        y=y,
        fourier_adaptive=True,
        max_adaptive_cols=1,
    )
    adaptive_srcs = {meta[c]["src"] for c in eng.columns if meta[c].get("adaptive") is True}
    assert adaptive_srcs.issubset({"f0"}), f"adaptive detection leaked past the cap: {adaptive_srcs}"


def test_max_adaptive_cols_unbounded_can_detect_beyond_first_column():
    """Sanity: without the cap, a later column CAN get adaptive detection (proves the cap in the
    previous test is genuinely constraining behavior, not just a coincidence of this fixture)."""
    n = 3000
    rng = np.random.default_rng(7)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(4)})
    y = np.sin(3.1 * X["f3"].to_numpy()) + 0.05 * rng.standard_normal(n)  # oscillatory in f3 (last col)
    eng, meta = generate_extra_basis_features(
        X,
        extra_bases=("fourier",),
        cols=list(X.columns),
        y=y,
        fourier_adaptive=True,
        max_adaptive_cols=None,
    )
    adaptive_srcs = {meta[c]["src"] for c in eng.columns if meta[c].get("adaptive") is True}
    assert "f3" in adaptive_srcs, "expected the unbounded detector to find f3's oscillation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
