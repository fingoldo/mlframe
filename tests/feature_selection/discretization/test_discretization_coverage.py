"""Additional coverage for discretization.py -- exercise each method-dispatch branch + edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import (
    discretize_array,
    discretize_uniform,
    discretize_2d_array,
    discretize_sklearn,
    categorize_dataset,
    digitize,
    get_binning_edges,
    edges,
)

try:
    from mlframe.feature_selection.filters.discretization import categorize_1d_array

    _HAVE_1D = True
except ImportError:
    _HAVE_1D = False


@pytest.mark.fast
def test_discretize_array_quantile():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(500).astype(np.float64)
    out = discretize_array(arr, n_bins=10, method="quantile")
    assert out.min() >= 0
    assert out.max() < 10
    assert len(out) == len(arr)


def test_discretize_array_uniform_via_method_dispatch():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal(500).astype(np.float64)
    out = discretize_array(arr, n_bins=10, method="uniform")
    assert out.min() >= 0
    assert out.max() < 10


def test_discretize_array_sklearn_dispatch():
    """``discretize_array`` no longer dispatches via a ``method='sklearn'``
    string (it only supports 'uniform' / 'quantile' now). The sklearn-
    based path lives on the dedicated ``discretize_sklearn`` helper -
    test goes there directly. The 'sklearn' string must now raise
    ValueError rather than silently fall through (pre-fix the test
    swallowed any exception via pytest.skip and ran no real assertion)."""
    rng = np.random.default_rng(2)
    arr = rng.standard_normal(500).astype(np.float64)
    with pytest.raises(ValueError, match="Unsupported discretization method"):
        discretize_array(arr, n_bins=10, method="sklearn")
    # Real coverage of the sklearn binning path via the dedicated helper.
    out = discretize_sklearn(arr, n_bins=10)
    assert out is not None
    assert out.shape == arr.shape


def test_discretize_uniform_direct():
    rng = np.random.default_rng(3)
    arr = rng.standard_normal(300).astype(np.float64)
    out = discretize_uniform(arr, n_bins=5)
    assert out.min() >= 0
    assert out.max() < 5


def test_discretize_2d_array_basic():
    rng = np.random.default_rng(4)
    arr = rng.standard_normal((200, 3)).astype(np.float64)
    out = discretize_2d_array(arr, n_bins=8, method="quantile")
    assert out.shape == arr.shape
    assert out.max() < 8


def test_discretize_2d_array_uniform():
    rng = np.random.default_rng(5)
    arr = rng.standard_normal((200, 3)).astype(np.float64)
    out = discretize_2d_array(arr, n_bins=8, method="uniform")
    assert out.shape == arr.shape


def test_discretize_sklearn_direct():
    rng = np.random.default_rng(6)
    arr = rng.standard_normal(300).astype(np.float64)
    out = discretize_sklearn(arr, n_bins=7)
    assert out is not None
    assert out.shape == arr.shape
    assert out.min() >= 0
    assert out.max() < 7


def test_categorize_dataset_smoke():
    """``categorize_dataset`` signature was simplified: the legacy
    ``quantization_nbins`` kwarg is now ``n_bins`` (matching the rest of
    the discretization API), and the ``categories_strategy`` kwarg was
    consolidated into ``method='quantile'`` / ``'uniform'``. Pre-fix the
    test swallowed the resulting TypeError via pytest.skip and never
    actually exercised the function."""
    import pandas as pd

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "num": rng.standard_normal(200),
            "cat": rng.choice(["A", "B", "C"], size=200),
            "int_ord": rng.integers(0, 5, 200),
        }
    )
    result = categorize_dataset(df, n_bins=5)
    assert result is not None


def test_digitize_smoke():
    arr = np.array([0.1, 0.5, 0.9, 1.5], dtype=np.float64)
    bin_edges = np.array([0.0, 0.4, 0.8, 1.2, 1.6], dtype=np.float64)
    out = digitize(arr, bin_edges)
    assert out.shape == arr.shape


def test_get_binning_edges_quantile():
    rng = np.random.default_rng(8)
    arr = rng.standard_normal(300).astype(np.float64)
    e = get_binning_edges(arr, n_bins=10, method="quantile")
    assert e is not None
    assert len(e) >= 2


def test_get_binning_edges_uniform():
    rng = np.random.default_rng(9)
    arr = rng.standard_normal(300).astype(np.float64)
    e = get_binning_edges(arr, n_bins=10, method="uniform")
    assert e is not None


def test_edges_helper():
    """edges(arr, quantiles) -> percentile-based edge array."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    quantiles = [0, 25, 50, 75, 100]
    e = edges(arr, quantiles)
    assert len(e) == len(quantiles)
    assert e[0] == 1.0
    assert e[-1] == 10.0


def test_discretize_array_constant_input():
    """All-same value: should not crash, all output bins are equal."""
    arr = np.full(100, 3.14, dtype=np.float64)
    out = discretize_array(arr, n_bins=5, method="quantile")
    # All bins land in same bucket
    assert len(np.unique(out)) == 1


def test_discretize_array_n_bins_two():
    """Minimum n_bins=2 path."""
    rng = np.random.default_rng(10)
    arr = rng.standard_normal(100).astype(np.float64)
    out = discretize_array(arr, n_bins=2, method="quantile")
    assert out.max() < 2


def test_discretize_array_n_bins_large():
    rng = np.random.default_rng(11)
    arr = rng.standard_normal(500).astype(np.float64)
    out = discretize_array(arr, n_bins=50, method="quantile")
    assert out.max() < 50


def test_categorize_1d_array_smoke():
    if not _HAVE_1D:
        # Helper isn't part of the public API in this build - skip is
        # appropriate (env-only, not a masked bug).
        pytest.skip("categorize_1d_array not exported")
    rng = np.random.default_rng(12)
    arr = rng.choice(["X", "Y", "Z"], 200)
    # ``categorize_1d_array`` requires explicit ``min_ncats``, ``method``,
    # ``astropy_sample_size`` and ``method_kwargs`` parameters. Pre-fix
    # the test wrapped this in a bare-except->skip that masked the
    # signature drift entirely; pass the smoke-defaults explicitly.
    out = categorize_1d_array(
        arr,
        min_ncats=50,
        method="quantile",
        astropy_sample_size=10_000,
        method_kwargs={},
    )
    assert out is not None


def test_discretize_array_unknown_method_raises_or_falls_back():
    """Unknown method string should raise or silently fall back to default."""
    rng = np.random.default_rng(13)
    arr = rng.standard_normal(100).astype(np.float64)
    try:
        out = discretize_array(arr, n_bins=5, method="nonexistent_method_xyz")
    except (ValueError, KeyError):
        return  # acceptable contract: explicit reject
    # Silent fallback is also acceptable; assert output is valid
    assert out is not None
