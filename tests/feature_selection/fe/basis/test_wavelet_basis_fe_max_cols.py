"""Regression tests for the wavelet held-out scale-selection column cap (2026-07-09 fix).

cProfile evidence (bench_mrmr_pre_categorize_family_profile.py, p=420/n=60000): the wavelet family's
held-out scale-selection (``_select_wavelet_legs``, internally calling ``_binned_mi`` per candidate
leg) is the second-largest default-ON pre-FE cost, ~26% of the whole pre-categorize wall -- roughly
linear in column count. ``max_cols`` bounds how many columns run this expensive selection; columns
beyond the cap get no wavelet legs at all (there is no cheap fallback the way Fourier has a fixed grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._wavelet_basis_fe import generate_wavelet_features


def _make_step_frame(n=1200, p=8, seed=0):
    """Each column carries an independent localized step in y (so wavelet legs should be admitted
    for any column the selector actually reaches)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.uniform(-1, 1, n) for i in range(p)})
    y = (X["f0"] > 0.3).astype(float) + 0.05 * rng.standard_normal(n)
    return X, y


def test_max_cols_none_preserves_legacy_behavior():
    X, y = _make_step_frame(p=5)
    eng_unbounded, _meta_unbounded = generate_wavelet_features(X, y=y, max_cols=None)
    eng_explicit_default, _meta_explicit_default = generate_wavelet_features(X, y=y)
    assert set(eng_unbounded.columns) == set(eng_explicit_default.columns)


def test_max_cols_zero_emits_no_wavelet_columns():
    X, y = _make_step_frame(p=5)
    eng, meta = generate_wavelet_features(X, y=y, max_cols=0)
    assert eng.empty
    assert meta == {}


def test_max_cols_bounds_which_columns_get_legs():
    """Only f0 carries a real localized step; with a cap of 1 (only column index 0 = 'f0' eligible),
    no OTHER column may ever get a wavelet leg regardless of its own structure."""
    n = 1200
    rng = np.random.default_rng(5)
    X = pd.DataFrame({f"f{i}": rng.uniform(-1, 1, n) for i in range(4)})
    # f0 AND f3 both carry independent localized steps -- without a cap both could get legs.
    y = (X["f0"] > 0.3).astype(float) + (X["f3"] < -0.3).astype(float) + 0.05 * rng.standard_normal(n)
    _eng, meta = generate_wavelet_features(X, y=y, cols=list(X.columns), max_cols=1)
    sources = {m["src"] for m in meta.values()}
    assert sources.issubset({"f0"}), f"wavelet leg leaked past the column cap: {sources}"


def test_max_cols_unbounded_can_select_a_later_column():
    """Sanity: without the cap, a later column CAN get legs (proves the cap above is genuinely
    constraining, not a coincidence of the fixture)."""
    n = 1200
    rng = np.random.default_rng(5)
    X = pd.DataFrame({f"f{i}": rng.uniform(-1, 1, n) for i in range(4)})
    y = (X["f3"] < -0.3).astype(float) + 0.05 * rng.standard_normal(n)
    _eng, meta = generate_wavelet_features(X, y=y, cols=list(X.columns), max_cols=None)
    sources = {m["src"] for m in meta.values()}
    assert "f3" in sources, "expected the unbounded selector to admit a leg on f3's localized step"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
