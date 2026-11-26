"""``sorted_quantiles`` also takes a double-precision vector whose values are all exactly float32 (a float32 column widened, the form the edge search reads its columns in): the radix route on the narrowed copy gives the quantiles, and the bin edges, of ``np.percentile``."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as A
from mlframe.feature_selection.filters.discretization._quantile_radix import sorted_quantiles

PCT = np.linspace(0.0, 100.0, 49)


def _column(n, seed, kind):
    """A float32-valued column of a kind the edge search meets, held in double precision."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    if kind == "ties":
        x = np.round(x, 1)
    elif kind == "both zeros":
        x[::5] = 0.0
        x[1::5] = -0.0
    elif kind == "few values":
        x = np.round(x * 2)
    elif kind == "heavy tail":
        x = rng.standard_cauchy(n).astype(np.float32)
    return x.astype(np.float64)


@pytest.mark.parametrize("kind", ["normal", "ties", "both zeros", "few values", "heavy tail"])
@pytest.mark.parametrize("n", [40_000, 120_001])
def test_the_radix_route_gives_the_quantiles_and_the_edges_of_numpy(kind, n):
    """The values equal ``np.percentile``'s (the zero's sign aside), so do the inner edges ``_edges_from_quantiles`` returns, for every kind of column and on two seeds."""
    for seed in (0, 1):
        x = _column(n, seed, kind)
        got = sorted_quantiles(x, PCT / 100.0)
        assert got is not None
        assert np.array_equal(got, np.percentile(x, PCT))
        legacy = np.unique(np.percentile(x, PCT))
        assert np.array_equal(A._edges_from_quantiles(x, 48), legacy[1:-1] if legacy.size > 2 else np.array([]))


def test_a_vector_that_is_not_exactly_float32_or_not_finite_or_short_goes_to_numpy():
    """Double values that float32 would round, a NaN, and a short vector decline the route (the caller then uses numpy)."""
    rng = np.random.default_rng(2)
    n = 60_000
    assert sorted_quantiles(rng.standard_normal(n), PCT / 100.0) is None  # a true double: not exactly a float32
    x = _column(n, 3, "normal")
    x[7] = np.nan
    assert sorted_quantiles(x, PCT / 100.0) is None
    assert sorted_quantiles(_column(500, 4, "normal"), PCT / 100.0) is None
    assert sorted_quantiles(np.arange(n, dtype=np.int64), PCT / 100.0) is None


def test_the_float32_route_is_unchanged():
    """A float32 vector takes the route it always did and gives the same numbers as its widened twin."""
    x32 = _column(50_000, 5, "normal").astype(np.float32)
    assert np.array_equal(sorted_quantiles(x32, PCT / 100.0), sorted_quantiles(x32.astype(np.float64), PCT / 100.0))


@pytest.mark.parametrize("kind", ["normal", "ties", "both zeros", "few values"])
@pytest.mark.parametrize("n", [60_000, 260_000])
def test_unique_with_counts_of_an_exact_float32_double_vector_equals_numpys(kind, n):
    """Above ``_NARROW_UNIQUE_MIN_SIZE`` the sort runs on the float32 copy: the distinct values (double precision, the zero's sign included up to what numpy's own ordering decides) and the counts are ``np.unique``'s."""
    from mlframe.feature_selection.filters.discretization._quantile_radix import unique_with_counts

    x = _column(n, 11, kind)
    values, counts = unique_with_counts(x)
    ref_values, ref_counts = np.unique(x, return_counts=True)
    assert values.dtype == np.float64 and counts.dtype == ref_counts.dtype
    assert np.array_equal(values, ref_values) and np.array_equal(counts, ref_counts)
