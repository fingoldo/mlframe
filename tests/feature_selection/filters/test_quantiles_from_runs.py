"""The quantile edges of a column are read off its distinct values and their counts (the edge search has them for every column) instead of sorting the column a second time, and they are the edges ``np.percentile`` gives."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as A
from mlframe.feature_selection.filters import _adaptive_nbins_columns as C
from mlframe.feature_selection.filters.discretization._quantile_radix import quantiles_from_runs, unique_with_counts

PCT = np.linspace(0.0, 100.0, 49)


def _column(n, seed, kind, dtype):
    """A column of a kind the edge search meets."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if kind == "ties":
        x = np.round(x, 1)
    elif kind == "few":
        x = np.round(x * 2)
    elif kind == "zeros":
        x[::3] = 0.0
        x[1::3] = -0.0
    elif kind == "heavy":
        x = rng.standard_cauchy(n)
    elif kind == "mostly one value":
        x = np.where(rng.random(n) < 0.97, 1.0, x)
    return x.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["normal", "ties", "few", "zeros", "heavy", "mostly one value"])
@pytest.mark.parametrize("n", [7, 2_000, 60_001])
def test_quantiles_from_runs_equal_numpys(dtype, kind, n):
    """For every kind of column, size and float width the quantiles are ``np.percentile``'s of the widened values, on two seeds."""
    for seed in (0, 1):
        x = _column(n, seed, kind, dtype)
        uniq, counts = unique_with_counts(x)
        got = quantiles_from_runs(uniq, counts, PCT / 100.0)
        assert got is not None
        assert np.array_equal(got, np.percentile(x.astype(np.float64), PCT))


def test_nothing_to_read_gives_none():
    """No runs, or runs and counts of different lengths, decline."""
    assert quantiles_from_runs(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64), PCT / 100.0) is None
    assert quantiles_from_runs(np.arange(3.0), np.ones(2, dtype=np.int64), PCT / 100.0) is None
    assert quantiles_from_runs(np.arange(3), np.ones(3, dtype=np.int64), PCT / 100.0) is None


@pytest.mark.parametrize("kind", ["normal", "ties", "zeros", "heavy"])
def test_the_edges_with_facts_equal_the_edges_without(kind):
    """``_edges_from_quantiles`` with the facts of the finite values gives the edges it gives without them, NaN and infinities in the column included; facts that do not add up to the column are ignored."""
    for n_bins in (3, 17, 48):
        x = _column(30_000, 5, kind, np.float64)
        x[::97] = np.nan
        x[1::193] = np.inf
        facts = unique_with_counts(x[np.isfinite(x)])
        assert np.array_equal(A._edges_from_quantiles(x, n_bins, facts), A._edges_from_quantiles(x, n_bins))
        wrong = (facts[0], facts[1] + 1)
        assert np.array_equal(A._edges_from_quantiles(x, n_bins, wrong), A._edges_from_quantiles(x, n_bins))


def test_per_feature_edges_reads_the_quantiles_off_the_facts_and_gets_the_same_edges(monkeypatch):
    """The partition scheme of ``per_feature_edges`` takes the route for long columns, and its edges equal those of a run with the route switched off."""
    rng = np.random.default_rng(7)
    n, p = 40_000, 12
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 3] = np.round(X[:, 3], 1)
    X[:, 5] = rng.integers(0, 6, n)
    calls = []
    real = C._an().quantiles_from_runs if hasattr(C._an(), "quantiles_from_runs") else A.quantiles_from_runs

    def spy(u, c, q):
        """Count the calls, answer as usual."""
        calls.append(int(u.shape[0]))
        return real(u, c, q)

    monkeypatch.setattr(A, "quantiles_from_runs", spy)
    C.clear_edge_memo()
    with_route = A.per_feature_edges(X, None, method="partition", partition="quantile", count_rule="rice_capped", n_jobs=2)
    assert calls, "the quantiles were not read off the facts"
    monkeypatch.setattr(A, "quantiles_from_runs", lambda u, c, q: None)
    C.clear_edge_memo()
    without = A.per_feature_edges(X, None, method="partition", partition="quantile", count_rule="rice_capped", n_jobs=2)
    assert len(with_route) == len(without) == p
    assert all(np.array_equal(a, b) for a, b in zip(with_route, without))
