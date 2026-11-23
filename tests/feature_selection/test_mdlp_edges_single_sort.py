"""MDLP column edges come from ONE sort per column: the distinct-value facts and the collapsed-column quantile fallback read the sorted column; edges equal the two-sort path."""
import numpy as np


def _frame(n=30000):
    """Columns that exercise every branch: signal, noise (collapsed to the quantile fallback), ties, low cardinality, NaN and inf, a float64 column."""
    rng = np.random.default_rng(11)
    y = (rng.random(n) < 0.4).astype(np.int8)
    sig = rng.standard_normal(n) + 1.5 * y
    cols = [
        sig.astype(np.float32), rng.standard_normal(n).astype(np.float32), np.round(rng.standard_normal(n), 1).astype(np.float32),
        (rng.random(n) < 0.01).astype(np.float32), np.where(rng.random(n) < 0.3, np.nan, sig).astype(np.float32),
        np.where(rng.random(n) < 0.05, np.inf, rng.standard_normal(n)).astype(np.float32), rng.standard_normal(n).astype(np.float64), sig.astype(np.float64),
    ]
    return np.column_stack([c.astype(np.float64) for c in cols]), y


def test_edges_equal_the_two_sort_path(monkeypatch):
    """per_feature_edges with the shared sort gives the same edges as with the presort switched off, column by column."""
    from mlframe.feature_selection.filters import _adaptive_nbins as A
    from mlframe.feature_selection.filters import _adaptive_nbins_columns as C

    X, y = _frame()
    C.clear_edge_memo()
    new = A.per_feature_edges(X, y, method="mdlp", n_jobs=1, max_adaptive_nbins=10)
    monkeypatch.setattr(C, "_presort_for_mdlp", lambda *a, **k: None)
    C.clear_edge_memo()
    old = A.per_feature_edges(X, y, method="mdlp", n_jobs=1, max_adaptive_nbins=10)
    assert len(new) == len(old) > 0, "no column was binned"
    for j, (a, b) in enumerate(zip(new, old)):
        assert a.shape == b.shape and np.array_equal(a, b), j


def test_sorted_helpers_match_numpy():
    """sorted_unique_with_counts equals np.unique and quantiles_of_sorted equals np.quantile on a sorted vector."""
    from mlframe.feature_selection.filters.discretization._quantile_radix import quantiles_of_sorted, sorted_unique_with_counts

    rng = np.random.default_rng(2)
    xs = np.sort(np.round(rng.standard_normal(5000), 2))
    u, c = sorted_unique_with_counts(xs)
    ru, rc = np.unique(xs, return_counts=True)
    assert np.array_equal(u, ru) and np.array_equal(c, rc)
    qs = np.linspace(0.0, 1.0, 6)
    assert np.array_equal(quantiles_of_sorted(xs, qs), np.quantile(xs, qs))
