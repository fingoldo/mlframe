"""The NaN guard must see non-finite values anywhere in the frame, and must treat +/-inf as missing.

TRC-07: the output check looked at the first 500 predictions only, and the input probe at the first 500 rows, so a
frame whose NaNs start further down (a lagged / rolling feature on a later entity) returned NaN predictions from a
NaN-tolerant model into metrics and ensemble stacks with no warning.
TRC-13: the probe and the imputer handled NaN only; +/-inf passed SimpleImputer unchanged and turned predictions into inf.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training._predict_guards import _apply_nan_guard, _frame_has_non_finite, prime_nan_guard_stats


def _frame(n=2000, seed=0):
    """A clean two-column frame long enough that row 1500 sits past any first-N sample."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})


def test_non_finite_values_past_the_first_500_rows_are_seen():
    """The guard scans the whole frame: a NaN at row 1500 and an inf at row 1800 both have to register."""
    X = _frame()
    assert not _frame_has_non_finite(X)
    X.loc[1500, "a"] = np.nan
    assert _frame_has_non_finite(X)
    Y = _frame()
    Y.loc[1800, "b"] = np.inf
    assert _frame_has_non_finite(Y)


def test_the_guard_imputes_a_late_nan_instead_of_passing_it_through():
    """Detecting the value is half of it; the row must also come back with a real prediction."""
    X = _frame()
    y = X["a"] * 2 + 1
    model = LinearRegression().fit(X, y)
    prime_nan_guard_stats(model, X)
    X_bad = X.copy()
    X_bad.loc[1500, "a"] = np.nan
    out = _apply_nan_guard(model, X_bad, model.predict, len(X_bad))
    assert np.isfinite(out).all(), "a NaN past row 500 must be imputed, not predicted through"


def test_an_infinite_input_is_imputed_like_a_nan():
    """An infinity is as unusable to a model as a NaN, so it takes the same imputation path."""
    X = _frame()
    y = X["a"] * 2 + 1
    model = LinearRegression().fit(X, y)
    prime_nan_guard_stats(model, X)
    X_bad = X.copy()
    X_bad.loc[10, "b"] = np.inf
    out = _apply_nan_guard(model, X_bad, model.predict, len(X_bad))
    assert np.isfinite(out).all()


def test_the_training_stats_ignore_an_infinite_training_value():
    """An inf in the priming frame must not poison the imputer's own statistics into inf."""
    X = _frame()
    X.loc[3, "a"] = np.inf
    model = LinearRegression().fit(_frame(), _frame()["a"])
    prime_nan_guard_stats(model, X)
    assert np.isfinite(model._mlframe_nan_imputer.statistics_).all()
    assert np.isfinite(model._mlframe_nan_scaler.mean_).all()
