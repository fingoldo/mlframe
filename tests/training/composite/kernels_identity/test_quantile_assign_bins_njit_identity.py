"""Regression: ``_quantile_residual_assign_bins`` njit parallel linear-scan kernel
must be bit-identical to the prior ``np.clip(np.searchsorted(...))`` reference,
including the NaN / +-inf / out-of-range edges.

Perf win measured in
``training/composite/transforms/_benchmarks/bench_quantile_assign_bins_searchsorted.py``
(3.9x@10k / 6.6x@200k / 8.9x@1M).
"""

import numpy as np
import pytest

import mlframe.training.composite.transforms.nonlinear as nl


def _reference_assign(base, edges):
    """The exact pre-optimization numpy implementation."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n_bins = edges.size - 1
    if n_bins <= 1:
        return np.zeros(base_f.size, dtype=np.intp)
    bin_idx = np.searchsorted(edges[1:-1], base_f, side="right")
    return np.clip(bin_idx, 0, n_bins - 1)


@pytest.mark.parametrize("n", [1, 7, 1000, 10_000, 200_000])
@pytest.mark.parametrize("n_bins", [2, 5, 10, 25])
def test_assign_bins_bit_identical_random(n, n_bins):
    """The numba quantile-residual bin-assign kernel is bit-identical to a pure-Python reference across n/n_bins sweeps."""
    rng = np.random.default_rng(n * 31 + n_bins)
    base = rng.standard_normal(n)
    edges = np.quantile(base, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        pytest.skip("degenerate edges")
    edges[0] = -np.inf
    edges[-1] = np.inf
    got = nl._quantile_residual_assign_bins(base, edges)
    exp = _reference_assign(base, edges)
    assert np.array_equal(got, exp)


def test_assign_bins_bit_identical_nan_inf_oob():
    """NaN -> top bin (searchsorted sorts NaN as +inf); +-inf and far-OOB clip."""
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf], dtype=np.float64)
    base = np.array(
        [np.nan, np.inf, -np.inf, 1e300, -1e300, 0.0, -0.5, 0.5, -1.0, 1.0],
        dtype=np.float64,
    )
    got = nl._quantile_residual_assign_bins(base, edges)
    exp = _reference_assign(base, edges)
    assert np.array_equal(got, exp)
    # NaN specifically maps to the last bin (n_bins-1 = 3)
    assert got[0] == edges.size - 2


def test_assign_bins_single_bin_degenerate():
    """A single-bin edge array (all values fall in one bucket) is handled identically to the reference, including NaN."""
    edges = np.array([-np.inf, np.inf], dtype=np.float64)
    base = np.array([np.nan, 1.0, -1.0, 0.0], dtype=np.float64)
    got = nl._quantile_residual_assign_bins(base, edges)
    assert np.array_equal(got, np.zeros(base.size, dtype=np.intp))
