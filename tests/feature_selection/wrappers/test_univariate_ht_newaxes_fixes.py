"""Regression tests for two bugs landed in the Wave-5 univariate-HT prescreen.

Covers:
  [7] _classify_target: high-cardinality integer regression targets were routed
      to 'multiclass' (Kruskal-Wallis) because the max(10, sqrt(n)) threshold
      grows with n. The cardinality-ratio guard (<= 0.05 * n) now sends them to
      'continuous'.
  [8] _chi2_sf scipy-free fallback: previously erfc(sqrt(x/2)) collapsed every df
      to df=1, producing wrong p-values for df >> 1. The df-aware fallback now
      uses Q(df/2, x/2) (regularized upper incomplete gamma).

These are focused unit tests of the smallest callables; no training run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_selection.wrappers import _univariate_ht as uht

# ---------------------------------------------------------------------------
# [7] _classify_target cardinality-ratio guard


def test_classify_high_cardinality_integer_regression_is_continuous():
    # The headline bug: 300 distinct integer values even on a huge dataset.
    # OLD threshold was max(10, sqrt(n)); at n=200_000 sqrt(n)~=447 >= 300 so the
    # 300-distinct-integer regression target was (wrongly) called 'multiclass' and
    # routed to Kruskal-Wallis. The absolute label cap (50) now sends it to
    # 'continuous' regardless of n.
    """Classify high cardinality integer regression is continuous."""
    rng = np.random.default_rng(0)
    n = 200_000
    y = rng.integers(0, 300, size=n).astype(np.int64)
    assert np.unique(y).size == 300
    assert uht._classify_target(y) == "continuous"

    # Also exercise the ratio guard directly: 300 unique over n=1000 (30% ratio).
    y2 = np.arange(1000, dtype=np.int64) % 300
    assert np.unique(y2).size == 300
    assert uht._classify_target(y2) == "continuous"


def test_classify_genuine_multiclass_still_multiclass():
    # 5 classes over many rows: low cardinality, ratio ~0 -> multiclass preserved.
    """Classify genuine multiclass still multiclass."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=5000).astype(np.int64)
    assert uht._classify_target(y) == "multiclass"


def test_classify_binary_unchanged():
    """Classify binary unchanged."""
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    assert uht._classify_target(y) == "binary"


def test_classify_float_integer_in_disguise_high_cardinality_is_continuous():
    # Float dtype but all-integer values, high cardinality relative to n -> continuous.
    """Classify float integer in disguise high cardinality is continuous."""
    n = 800
    y = np.arange(n, dtype=np.float64) % 200  # 200 distinct, 25% ratio
    assert np.unique(y).size == 200
    assert uht._classify_target(y) == "continuous"


def test_is_multiclass_cardinality_helper_boundaries():
    # Below all caps -> multiclass.
    """Is multiclass cardinality helper boundaries."""
    assert uht._is_multiclass_cardinality(5, 5000) is True
    # Exceeds the 5% ratio cap -> not multiclass even if below sqrt(n).
    assert uht._is_multiclass_cardinality(300, 1000) is False
    # Exceeds the absolute label cap (50) even with a tiny ratio -> not multiclass.
    assert uht._is_multiclass_cardinality(300, 1_000_000) is False
    # Exactly at the absolute floor of 10 with tiny ratio -> multiclass.
    assert uht._is_multiclass_cardinality(10, 10_000) is True


# ---------------------------------------------------------------------------
# [8] df-aware chi-squared survival-function fallback


def test_regularized_upper_gamma_matches_scipy_chi2_sf_for_large_df():
    """Regularized upper gamma matches scipy chi2 sf for large df."""
    scipy_stats = pytest.importorskip("scipy.stats")
    chi2 = scipy_stats.chi2
    # Exercise df well above 1, where the old erfc(sqrt(x/2)) collapse was wrong.
    for df in (1, 2, 5, 10, 50, 99):
        for x in (0.5, 2.0, df * 1.0, df * 2.0, df * 3.0):
            got = uht._regularized_upper_gamma_q(df / 2.0, x / 2.0)
            ref = float(chi2.sf(x, df))
            assert abs(got - ref) < 1e-6, f"df={df} x={x}: got {got} ref {ref}"


def test_chi2_sf_fallback_is_df_aware_not_df1_collapse(monkeypatch):
    # Force the scipy-free fallback branch by making the scipy import fail, then
    # confirm the fallback differs from the OLD df=1 formula for df>1.
    """Chi2 sf fallback is df aware not df1 collapse."""
    scipy_stats = pytest.importorskip("scipy.stats")
    chi2_ref = scipy_stats.chi2

    import builtins

    real_import = builtins.__import__

    def _no_scipy_stats(name, *args, **kwargs):
        """No scipy stats."""
        if name == "scipy.stats" or name.startswith("scipy.stats"):
            raise ImportError("forced: no scipy for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scipy_stats)

    x, df = 30.0, 10
    old_df1_formula = float(math.erfc(math.sqrt(x / 2.0)))  # the buggy collapse
    fallback = uht._chi2_sf(x, df)
    true_sf = float(chi2_ref.sf(x, df))  # chi2_ref captured before monkeypatch

    # Fallback tracks the true df-aware SF, not the df=1 collapse.
    assert abs(fallback - true_sf) < 1e-6
    # And it is materially different from the old (wrong) df=1 value: the buggy
    # collapse underestimates the true SF by ~4 orders of magnitude here.
    assert fallback > 10.0 * old_df1_formula


def test_chi2_sf_degenerate_inputs_return_one():
    """Chi2 sf degenerate inputs return one."""
    assert uht._chi2_sf(float("nan"), 5) == 1.0
    assert uht._chi2_sf(-1.0, 5) == 1.0
    assert uht._chi2_sf(5.0, 0) == 1.0
