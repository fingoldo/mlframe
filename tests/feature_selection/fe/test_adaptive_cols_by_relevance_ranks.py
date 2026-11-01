"""The adaptive-column relevance ranking uses the inverse permutation as the rank vector and keeps the choice of columns."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import _adaptive_cols_by_relevance


def test_selection_equals_double_argsort_reference():
    """Ties included: the chosen set equals the one the double-argsort scores give."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(8)})
    X["tied"] = np.round(rng.normal(size=n), 1)
    y = X["c3"].to_numpy() ** 2 + 0.5 * X["tied"].to_numpy() + rng.normal(size=n)
    cols = list(X.columns)
    got = _adaptive_cols_by_relevance(X, cols, y, 3)
    yy = np.asarray(y, dtype=np.float32 if X["c0"].dtype == np.float32 else np.float64)
    yy = yy - yy.mean()
    scores = {}
    for c in cols:
        x = X[c].to_numpy()
        bins = np.minimum(np.argsort(np.argsort(x, kind="stable"), kind="stable") * 64 // len(x), 63)
        cnt = np.bincount(bins, minlength=64)
        sm = np.bincount(bins, weights=yy, minlength=64)
        scores[c] = float(np.sum(np.where(cnt > 0, sm * sm / np.maximum(cnt, 1), 0.0)))
    assert got == set(sorted(scores, key=lambda c: -scores[c])[:3])
