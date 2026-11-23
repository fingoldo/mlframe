"""``MRMR.target_transform_`` (PLAN_target_transform_in_mrmr, step 3): metadata only, off by default, never changes the selection or the target.

The business-value check is on the ORIGINAL scale: with the accepted transform a ridge model has a lower held-out RMSE than without it (the quantity the gate protects), and on a linear target the verdict is the
identity (the false positive of the proposer is rejected).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.feature_selection.filters._target_transform_search import default_transforms
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(kind: str, n: int = 6000, seed: int = 0):
    """Eight numeric columns and a cubic or linear target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"x{i}" for i in range(8)])
    noise = rng.standard_normal(n)
    if kind == "cube":
        y = (X["x0"] + 0.7 * X["x1"] + 0.3 * noise) ** 3
    else:
        y = X["x0"] + 0.7 * X["x1"] + 0.3 * noise
    return X, y.to_numpy()


def _fit(X, y, **kw):
    """A fit without feature engineering (the metadata does not depend on it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=0, fe_max_steps=0, **kw).fit(X, y)


def test_off_by_default_leaves_the_attribute_none():
    """The default is off and the attribute exists."""
    X, y = _frame("cube")
    assert MRMR().target_transform_search is False
    assert _fit(X, y).target_transform_ is None


def test_cubic_target_gets_cbrt_and_the_original_scale_error_falls():
    """The verdict is cbrt, accepted, and a ridge on cbrt(y) (inverted) beats a ridge on y on held-out rows."""
    X, y = _frame("cube")
    verdict = _fit(X, y, target_transform_search=True).target_transform_
    assert verdict["name"] == "cbrt" and verdict["accepted"] is True
    tr = np.arange(len(y)) < len(y) // 2
    t = next(t for t in default_transforms() if t.name == verdict["name"])
    plain = Ridge(1.0).fit(X[tr], y[tr]).predict(X[~tr])
    warped = t.inverse(Ridge(1.0).fit(X[tr], t.forward(y[tr])).predict(X[~tr]))
    assert np.sqrt(np.mean((warped - y[~tr]) ** 2)) < 0.8 * np.sqrt(np.mean((plain - y[~tr]) ** 2))


def test_linear_target_gets_the_identity():
    """The proposer's signed_sqrt false positive is rejected by the gate: not accepted."""
    X, y = _frame("linear")
    verdict = _fit(X, y, target_transform_search=True).target_transform_
    assert verdict["name"] == "identity" and verdict["accepted"] is False
    assert verdict["proposals"][0][0] == "signed_sqrt"


def test_the_selection_and_the_target_are_unchanged_by_the_flag():
    """Same selected names with the flag on and off; the target array the caller holds is untouched."""
    X, y = _frame("cube")
    y_before = y.copy()
    on = _fit(X, y, target_transform_search=True)
    off = _fit(X, y)
    assert list(on.get_feature_names_out()) == list(off.get_feature_names_out())
    assert np.array_equal(y, y_before)


def test_a_class_target_is_skipped():
    """A binary target has no scale to transform: the verdict stays None even with the flag on."""
    X, y = _frame("linear")
    assert _fit(X, (y > 0).astype(int), target_transform_search=True).target_transform_ is None
