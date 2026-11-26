"""Cluster 20 regression: missing values get a split-gain bin of their own, apart from the highest finite values (audit 2026-11-20, 20-F7)."""

import numpy as np

from mlframe.reporting.charts._split_gain_ranking import _column_codes, split_gain_per_feature


def test_nan_rows_have_a_code_above_every_finite_bin():
    """NaN rows no longer share a bin with the largest finite values (20-F7)."""
    col = np.concatenate([np.linspace(0, 1, 200), np.full(20, np.nan)])
    codes = _column_codes(col, 8)
    assert codes[200:].min() > codes[:200].max()
    assert len(set(codes[200:].tolist())) == 1


def test_missingness_carrying_the_error_is_found_without_diluting_the_top_bin():
    """The error is high exactly on the NaN rows and flat elsewhere: the gain is large for that column and ~0 for a noise column (20-F7)."""
    rng = np.random.default_rng(0)
    n = 2000
    col = rng.normal(size=n)
    miss = rng.uniform(size=n) < 0.1
    col[miss] = np.nan
    err = np.where(miss, 1.0, 0.0) + 0.01 * rng.normal(size=n)
    noise = rng.normal(size=n)
    gains = split_gain_per_feature(np.column_stack([col, noise]), err)
    assert gains[0] > 20 * gains[1]
