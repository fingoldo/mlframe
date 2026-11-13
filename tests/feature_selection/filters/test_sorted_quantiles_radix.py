"""``sorted_quantiles`` (radix sort of a float32 copy plus numpy's linear interpolation in float64) equals ``np.nanpercentile`` of the float64-widened column bit for bit, and ``edges_qs`` returns the same edges as before.

The quantile spacing binner widened every column to float64 and called ``np.nanpercentile`` (a partition, 4.4 ms for 100000 values); the radix route takes 1 ms. The quantile grids of ``edges_qs`` (3 to 64 bins), tied and constant columns, zeros of both signs,
float32 extremes and sizes around the switch-over must give identical float64 values; other dtypes and small vectors return ``None`` so the caller keeps numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as ab
from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


def _grid(n_q: int) -> "tuple[np.ndarray, np.ndarray]":
    """The percentile grid ``edges_qs`` builds and the fractions it hands to the radix route (``q_pos / 100``, which numpy divides the same way)."""
    q_pos = np.linspace(0.0, 100.0, n_q + 1)
    return q_pos, q_pos / 100.0


def _reference(values: np.ndarray, q_pos: np.ndarray) -> np.ndarray:
    """What ``edges_qs`` computed before: ``nanpercentile`` of the float64-widened finite values."""
    return np.nanpercentile(values.astype(np.float64), q_pos)


@pytest.mark.parametrize("n", [4096, 5000, 100_003])
@pytest.mark.parametrize("kind", ["continuous", "ties", "ten_levels", "zeros", "constant"])
def test_quantiles_equal_numpy_bit_for_bit(n, kind):
    """The grids edges_qs uses for 3 to 64 bins on columns of several shapes."""
    rng = np.random.default_rng(n + len(kind))
    if kind == "continuous":
        x = rng.standard_normal(n).astype(np.float32)
    elif kind == "ties":
        x = rng.integers(0, 3000, n).astype(np.float32)
    elif kind == "ten_levels":
        x = rng.integers(0, 10, n).astype(np.float32)
    elif kind == "zeros":
        x = np.where(rng.random(n) < 0.4, np.float32(0.0), rng.standard_normal(n).astype(np.float32))
        x[rng.random(n) < 0.1] = np.float32(-0.0)
    else:
        x = np.full(n, 2.5, dtype=np.float32)
    for n_q in (3, 4, 7, 10, 33, 64):
        q_pos, fractions = _grid(n_q)
        got = qr.sorted_quantiles(x, fractions)
        assert got is not None and got.dtype == np.float64
        np.testing.assert_array_equal(got, _reference(x, q_pos))


def test_float32_extremes_and_two_values():
    """The largest and smallest finite float32 values and a two-level column."""
    n = 8192
    info = np.finfo(np.float32)
    x = np.where(np.arange(n) % 3 == 0, info.max, np.where(np.arange(n) % 3 == 1, info.min, np.float32(0.0))).astype(np.float32)
    two = np.where(np.arange(n) % 2 == 0, np.float32(1.0), np.float32(2.0))
    q_pos, fractions = _grid(10)
    for col in (x, two):
        np.testing.assert_array_equal(qr.sorted_quantiles(col, fractions), _reference(col, q_pos))


def test_other_dtypes_and_small_vectors_return_none():
    """float64, int, 2-D and vectors under the switch-over size are left to numpy."""
    rng = np.random.default_rng(1)
    fractions = np.linspace(0.0, 1.0, 5)
    assert qr.sorted_quantiles(rng.standard_normal(10_000), fractions) is None
    assert qr.sorted_quantiles(rng.integers(0, 9, 10_000).astype(np.int32), fractions) is None
    assert qr.sorted_quantiles(rng.standard_normal(100).astype(np.float32), fractions) is None
    assert qr.sorted_quantiles(rng.standard_normal((200, 50)).astype(np.float32), fractions) is None


@pytest.mark.parametrize("alpha", [0.25, 0.30, 0.35])
@pytest.mark.parametrize("n", [5000, 100_000])
def test_edges_qs_is_unchanged_for_float32_and_float64_columns(alpha, n, monkeypatch):
    """The binner's edges for a float32 column (radix route) equal the edges of the same values taken through the numpy route, and a float64 column is untouched."""
    rng = np.random.default_rng(int(alpha * 100) + n)
    x32 = (rng.standard_normal(n) * 3).astype(np.float32)
    x32[rng.random(n) < 0.02] = np.nan
    fast = ab.edges_qs(x32, alpha)
    monkeypatch.setattr(ab, "sorted_quantiles", lambda values, quantiles: None)
    slow = ab.edges_qs(x32, alpha)
    np.testing.assert_array_equal(fast, slow)
    assert fast.dtype == np.float64 and fast.size > 0
    np.testing.assert_array_equal(ab.edges_qs(x32.astype(np.float64), alpha), slow)


@pytest.mark.parametrize("n_bins", [2, 3, 5, 10, 33, 64])
@pytest.mark.parametrize("kind", ["continuous", "ties", "constant", "zeros", "with_nonfinite"])
def test_edges_from_quantiles_is_unchanged_for_float32_columns(n_bins, kind, monkeypatch):
    """The fixed-bin quantile edges of a float32 column (radix route, NaN and inf dropped first) equal the numpy route's; a float64 column is untouched."""
    rng = np.random.default_rng(n_bins + len(kind))
    n = 20_000
    if kind == "continuous":
        x = rng.standard_normal(n).astype(np.float32)
    elif kind == "ties":
        x = rng.integers(0, 40, n).astype(np.float32)
    elif kind == "constant":
        x = np.full(n, 1.5, dtype=np.float32)
    elif kind == "zeros":
        x = np.where(rng.random(n) < 0.6, np.float32(0.0), rng.standard_normal(n).astype(np.float32))
    else:
        x = rng.standard_normal(n).astype(np.float32)
        x[rng.random(n) < 0.03] = np.nan
        x[rng.random(n) < 0.01] = np.inf
        x[rng.random(n) < 0.01] = -np.inf
    fast = ab._edges_from_quantiles(x, n_bins)
    monkeypatch.setattr(ab, "sorted_quantiles", lambda values, quantiles: None)
    slow = ab._edges_from_quantiles(x, n_bins)
    np.testing.assert_array_equal(fast, slow)
    np.testing.assert_array_equal(ab._edges_from_quantiles(x.astype(np.float64), n_bins), slow)
