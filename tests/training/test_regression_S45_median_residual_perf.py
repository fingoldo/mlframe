"""Regression sensor S45 (Wave 4b): verify that the size-aware dispatcher
for the per-bin reduction in ``_median_residual_fit`` and
``_quantile_residual_fit`` returns numerically-identical results to the
reference v1 pyloop across the routed size classes.

The dispatcher routes to a pandas-groupby variant on small n (<=200k) and
falls back to the numpy mask-loop on larger n. Both variants MUST agree
bit-for-bit with the pre-dispatch reference implementation; otherwise
discovery / scoring on composite targets silently changes value.

A perf assertion is intentionally NOT included -- the measured speedup is
modest (1.05-1.45x) and platform-noise-dominated; the numerical-identity
sensor is the load-bearing guarantee.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_inputs(n: int, n_bins: int, seed: int = 17):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float64)
    base = rng.standard_normal(n).astype(np.float64)
    edges = np.quantile(base, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    n_bins_eff = edges.size - 1
    bin_idx = np.clip(
        np.searchsorted(edges[1:-1], base, side="right"), 0, n_bins_eff - 1,
    )
    return y, bin_idx.astype(np.intp), n_bins_eff


@pytest.mark.parametrize("n,n_bins", [
    (1_000, 5), (10_000, 10), (100_000, 20),
    (250_000, 10), (250_000, 20),
])
def test_median_residual_per_bin_medians_dispatcher_matches_v1(n, n_bins):
    """Dispatcher output == v1 reference at every routed size class. ``rtol=1e-12`` because median is exact on float64."""
    from mlframe.training.composite_transforms import (
        _median_residual_per_bin_medians,
        _median_residual_per_bin_medians_v1_pyloop,
    )
    y, bin_idx, n_bins_eff = _make_inputs(n, n_bins)
    ref = _median_residual_per_bin_medians_v1_pyloop(y, bin_idx, n_bins_eff)
    got = _median_residual_per_bin_medians(y, bin_idx, n_bins_eff)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("n,n_bins", [
    (1_000, 5), (10_000, 10), (100_000, 20),
    (250_000, 10), (250_000, 20),
])
def test_median_residual_pandas_variant_matches_v1(n, n_bins):
    """Explicit v2 (pandas groupby) variant -- numerical identity vs v1."""
    from mlframe.training.composite_transforms import (
        _median_residual_per_bin_medians_v1_pyloop,
        _median_residual_per_bin_medians_v2_pandas_groupby,
    )
    y, bin_idx, n_bins_eff = _make_inputs(n, n_bins)
    ref = _median_residual_per_bin_medians_v1_pyloop(y, bin_idx, n_bins_eff)
    got = _median_residual_per_bin_medians_v2_pandas_groupby(y, bin_idx, n_bins_eff)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_median_residual_per_bin_empty_bin_uses_global_median():
    """When a bin has zero rows, both variants fall back to ``np.median(y)``."""
    from mlframe.training.composite_transforms import (
        _median_residual_per_bin_medians,
        _median_residual_per_bin_medians_v1_pyloop,
        _median_residual_per_bin_medians_v2_pandas_groupby,
    )
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500).astype(np.float64)
    bin_idx = np.zeros(500, dtype=np.intp)
    bin_idx[:100] = 2
    bin_idx[100:] = 4
    n_bins = 5
    ref = _median_residual_per_bin_medians_v1_pyloop(y, bin_idx, n_bins)
    got1 = _median_residual_per_bin_medians(y, bin_idx, n_bins)
    got2 = _median_residual_per_bin_medians_v2_pandas_groupby(y, bin_idx, n_bins)
    np.testing.assert_allclose(got1, ref, rtol=1e-12)
    np.testing.assert_allclose(got2, ref, rtol=1e-12)
    expected_global = float(np.median(y))
    for empty_b in (0, 1, 3):
        assert ref[empty_b] == pytest.approx(expected_global, rel=1e-12)
        assert got1[empty_b] == pytest.approx(expected_global, rel=1e-12)


@pytest.mark.parametrize("n,n_bins,min_bin_n", [
    (5_000, 10, 50), (50_000, 10, 50), (250_000, 20, 100),
])
def test_quantile_residual_per_bin_stats_dispatcher_matches_v1(n, n_bins, min_bin_n):
    """Dispatcher for quantile-residual per-bin stats agrees with the v1 mask-loop bit-for-bit."""
    from mlframe.training._composite_transforms_nonlinear import (
        _quantile_residual_per_bin_stats,
        _quantile_residual_per_bin_stats_v1_pyloop,
    )
    y, bin_idx, n_bins_eff = _make_inputs(n, n_bins)
    global_median = float(np.median(y))
    global_iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-6)
    ref_med, ref_iqr, ref_sz = _quantile_residual_per_bin_stats_v1_pyloop(
        y, bin_idx, n_bins_eff, min_bin_n, global_median, global_iqr,
    )
    got_med, got_iqr, got_sz = _quantile_residual_per_bin_stats(
        y, bin_idx, n_bins_eff, min_bin_n, global_median, global_iqr,
    )
    np.testing.assert_allclose(got_med, ref_med, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(got_iqr, ref_iqr, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(got_sz, ref_sz)


def test_quantile_residual_per_bin_under_populated_bin_falls_back_to_global():
    """Bins with count < ``min_bin_n`` keep the global median / IQR fallback in BOTH variants. This is the load-bearing semantics carve-out: pandas groupby naturally returns all bins; we must reset under-populated rows."""
    from mlframe.training._composite_transforms_nonlinear import (
        _quantile_residual_per_bin_stats_v1_pyloop,
        _quantile_residual_per_bin_stats_v2_pandas_groupby,
    )
    rng = np.random.default_rng(0)
    n_bins = 5
    n = 2_000
    y = rng.standard_normal(n).astype(np.float64)
    bin_idx = rng.integers(0, n_bins, size=n).astype(np.intp)
    bin_idx[:5] = 0
    bin_idx[5:n - 5] = rng.integers(1, n_bins, size=n - 10)
    min_bin_n = 50
    global_median = float(np.median(y))
    global_iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-6)
    ref_med, ref_iqr, ref_sz = _quantile_residual_per_bin_stats_v1_pyloop(
        y, bin_idx, n_bins, min_bin_n, global_median, global_iqr,
    )
    got_med, got_iqr, got_sz = _quantile_residual_per_bin_stats_v2_pandas_groupby(
        y, bin_idx, n_bins, min_bin_n, global_median, global_iqr,
    )
    np.testing.assert_allclose(got_med, ref_med, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(got_iqr, ref_iqr, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(got_sz, ref_sz)


def test_median_residual_fit_end_to_end_matches_pre_dispatch_semantics():
    """End-to-end check: ``_median_residual_fit`` output schema + values match what the pre-dispatch pyloop produced. Uses the v1 helper to reconstruct the expected ``bin_medians`` array directly."""
    from mlframe.training.composite_transforms import (
        _median_residual_fit,
        _median_residual_per_bin_medians_v1_pyloop,
        _MEDIAN_RESIDUAL_N_BINS,
    )
    rng = np.random.default_rng(42)
    y = rng.standard_normal(5_000).astype(np.float64)
    base = rng.standard_normal(5_000).astype(np.float64)
    params = _median_residual_fit(y, base)
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    bin_idx = np.digitize(base, edges[1:-1])
    expected = _median_residual_per_bin_medians_v1_pyloop(y, bin_idx, edges.size - 1)
    np.testing.assert_allclose(
        np.asarray(params["bin_medians"]), expected, rtol=1e-12, atol=0.0,
    )


def test_quantile_residual_fit_end_to_end_matches_pre_dispatch_semantics():
    """End-to-end equivalent for ``_quantile_residual_fit``: ``bin_medians`` / ``bin_iqrs`` / ``bin_sizes`` must match the pre-dispatch mask-loop."""
    from mlframe.training._composite_transforms_nonlinear import (
        _quantile_residual_fit,
        _quantile_residual_per_bin_stats_v1_pyloop,
    )
    rng = np.random.default_rng(7)
    y = rng.standard_normal(10_000).astype(np.float64)
    base = rng.standard_normal(10_000).astype(np.float64)
    params = _quantile_residual_fit(y, base)
    if int(params.get("n_bins", 0)) <= 1:
        pytest.skip("degenerate single-bin path -- separate code branch")
    edges = np.asarray(params["bin_edges"], dtype=np.float64)
    actual_n_bins = edges.size - 1
    bin_idx = np.clip(
        np.searchsorted(edges[1:-1], base, side="right"), 0, actual_n_bins - 1,
    )
    finite = np.isfinite(y) & np.isfinite(base)
    y_clean = y[finite]
    base_clean = base[finite]
    bin_idx_clean = np.clip(
        np.searchsorted(edges[1:-1], base_clean, side="right"), 0, actual_n_bins - 1,
    )
    global_median = float(params["global_median"])
    global_iqr = float(params["global_iqr"])
    ref_med, ref_iqr, ref_sz = _quantile_residual_per_bin_stats_v1_pyloop(
        y_clean, bin_idx_clean, actual_n_bins, 50, global_median, global_iqr,
    )
    np.testing.assert_allclose(
        np.asarray(params["bin_medians"]), ref_med, rtol=1e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(params["bin_iqrs"]), ref_iqr, rtol=1e-12, atol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(params["bin_sizes"]), ref_sz)
