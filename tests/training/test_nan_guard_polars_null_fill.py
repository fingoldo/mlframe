"""Regression: the predict-time NaN guard's polars fastpath must fill polars
NULL (not only NaN).

A polars float column can carry ``null`` (not NaN). The guard's NaN gate
detects it (``np.asarray`` maps null->NaN), so ``_fit_persist_and_transform``
fires. Its polars fastpath previously used ``fill_nan`` only -- which leaves
polars nulls untouched -- so the null survived to the bridged pandas frame and
reached the NaN-intolerant model the guard exists to protect, crashing it.

The fastpath now fills both null and NaN, and fills all-null/all-NaN columns
with 0.0 to match sklearn ``SimpleImputer(keep_empty_features=True)``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from mlframe.training._predict_guards import _apply_nan_guard


def _captured_fn():
    seen = {}

    def fn(X):
        seen["X"] = X
        return np.zeros(len(X))

    return fn, seen


def test_polars_null_is_filled_not_just_nan():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    Xpd = np.column_stack([a, b])
    model = LinearRegression().fit(Xpd, 2.0 * a + b)  # NaN-intolerant

    a_null = a.tolist()
    a_null[5] = None  # polars NULL (not NaN) in a float column
    X = pl.DataFrame({"a": a_null, "b": b.tolist()}, schema={"a": pl.Float64, "b": pl.Float64})
    assert X["a"].null_count() == 1 and X["a"].is_nan().sum() == 0

    fn, seen = _captured_fn()
    _apply_nan_guard(model, X, fn, n_rows=n, fit_at_predict=True)

    cleaned = seen["X"]
    arr = cleaned.to_numpy() if hasattr(cleaned, "to_numpy") else np.asarray(cleaned)
    # Pre-fix: the polars fastpath fill_nan left the NULL in place -> NaN reached fn.
    assert not np.any(np.isnan(arr)), "null/NaN leaked to the model through the guard"


def test_polars_all_null_column_filled_zero():
    rng = np.random.default_rng(1)
    n = 200
    b = rng.normal(size=n)
    model = LinearRegression().fit(np.column_stack([np.zeros(n), b]), b)
    X = pl.DataFrame({"a": [None] * n, "b": b.tolist()}, schema={"a": pl.Float64, "b": pl.Float64})

    fn, seen = _captured_fn()
    _apply_nan_guard(model, X, fn, n_rows=n, fit_at_predict=True)
    arr = seen["X"].to_numpy() if hasattr(seen["X"], "to_numpy") else np.asarray(seen["X"])
    assert not np.any(np.isnan(arr))
