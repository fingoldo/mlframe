"""Regression: the predict-time NaN guard must keep all-NaN columns (not drop them).

A frame can carry a fully-NaN feature (e.g. MRMR emits a NaN column for a recipe whose engineered source was
pruned at fit). sklearn SimpleImputer drops all-NaN columns by default (keep_empty_features=False), so the guard's
``imputer.transform`` returned one fewer column than ``X.columns`` -> ``pd.DataFrame(values_(k-1), columns=k_names)``
raised "Shape of passed values is (N, k-1), indices imply (N, k)". The guard now uses keep_empty_features=True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training._predict_guards import _apply_nan_guard


def test_nan_guard_keeps_all_nan_column():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = 2.0 * X["a"] + X["b"]
    model = LinearRegression().fit(X, y)  # NaN-intolerant -> guard fires when X has NaN

    X_pred = X.copy()
    X_pred["c"] = np.nan  # whole column NaN (the MRMR unresolved-recipe shape)

    # Pre-fix: imputer dropped 'c' -> (n, 2) values vs (n, 3) names -> ValueError. Must not raise now.
    preds = _apply_nan_guard(model, X_pred, model.predict, n_rows=n, fit_at_predict=True)
    assert np.asarray(preds).shape[0] == n
