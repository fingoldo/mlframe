"""Row-statistic operator: the pool ranking never reads the held-out rows and an infinite value does not blank a column's standardisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._row_stat_fe import MAX_POOL, _pool_columns, row_stat_candidates


def test_pool_ranking_does_not_read_held_out_rows():
    """Changing only the odd-row target codes (the held-out evidence of the finalists) leaves the pool ranking unchanged."""
    n = 4000
    r = np.random.default_rng(0)
    cols = [f"x{i}" for i in range(MAX_POOL + 4)]
    X = pd.DataFrame(r.standard_normal((n, len(cols))), columns=cols)
    rows = np.arange(n)
    even = rows % 2 == 0
    codes_a = r.integers(0, 3, n).astype(np.int64)
    codes_b = codes_a.copy()
    last = X[cols[-1]].to_numpy()
    codes_b[~even] = np.digitize(last, np.quantile(last, [1 / 3, 2 / 3]))[~even]
    assert _pool_columns(X, cols, rows, codes_a, 3) == _pool_columns(X, cols, rows, codes_b, 3)


def test_infinite_value_does_not_blank_the_standardisation():
    """One +inf in a source column leaves the frozen mean and sd of every finalist finite."""
    n = 4000
    r = np.random.default_rng(0)
    X = pd.DataFrame(r.standard_normal((n, 10)), columns=[f"x{i}" for i in range(10)])
    A = X[["x1", "x3", "x5", "x6"]].to_numpy()
    y = (A.max(1) - A.min(1)) + 0.3 * r.standard_normal(n)
    X.loc[5, "x1"] = np.inf
    cands = row_stat_candidates(X, y, list(X.columns))
    assert any("x1" in c["src"] for c in cands), "the column holding the inf must still take part in the subset search"
    for c in cands:
        assert np.isfinite(c["mu"]).all() and np.isfinite(c["sd"]).all()
