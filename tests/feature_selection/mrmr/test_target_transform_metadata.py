"""``MRMR.target_transform_`` (PLAN_target_transform_in_mrmr, step 3): metadata only, off by default, never changes the selection or the target.

The business-value check is on the ORIGINAL scale: with the accepted transform a ridge model has a lower held-out MAE than without it (the quantity the gate protects), and on a linear target the verdict is the
identity (the false positive of the proposer is rejected).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
    """The verdict is cbrt, accepted, and a ridge on cbrt(y) (inverted with the stored transform) beats a ridge on y on held-out rows by MAE."""
    X, y = _frame("cube")
    verdict = _fit(X, y, target_transform_search=True).target_transform_
    assert verdict["name"] == "cbrt" and verdict["accepted"] is True and verdict["metric"] == "mae"
    assert verdict["err_chosen"] < 0.8 * verdict["err_identity"]
    tr = np.arange(len(y)) < len(y) // 2
    t = verdict["transform"]
    plain = Ridge(1.0).fit(X[tr], y[tr]).predict(X[~tr])
    warped = t.inverse(Ridge(1.0).fit(X[tr], t.forward(y[tr])).predict(X[~tr]))
    assert np.mean(np.abs(warped - y[~tr])) < 0.8 * np.mean(np.abs(plain - y[~tr]))


def test_linear_target_gets_the_identity():
    """A linear target is not transformed: the verdict is the identity with a reason."""
    X, y = _frame("linear")
    verdict = _fit(X, y, target_transform_search=True).target_transform_
    assert verdict["name"] == "identity" and verdict["accepted"] is False and verdict["reason"]


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


def test_the_link_and_the_unary_forms_of_the_inputs_are_both_named():
    """``y = exp(0.5 (sin(x1) + log(x2) + 0.5 x3))``: the link ``log`` is found with the additive-spline head, and the unary credit run on the linked target names ``sin`` for x1 and ``log`` for x2 (on the raw
    target it names ``sin`` and ``sqrt``, an approximation of the multiplicative structure). A ridge head on the named forms and the link beats a ridge on the raw columns by MAE on fresh rows."""
    n = 6000

    def frame(seed):
        """The data of the case."""
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({"x1": rng.uniform(-6, 6, n), "x2": rng.lognormal(0, 1, n), "x3": rng.standard_normal(n), "n0": rng.standard_normal(n), "n1": rng.standard_normal(n)})
        index = 0.5 * (np.sin(X["x1"]) + np.log(X["x2"]) + 0.5 * X["x3"]) + 0.05 * rng.standard_normal(n)
        return X, np.exp(index.to_numpy())

    X, y = frame(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=1, target_transform_search=True).fit(X, y)
    verdict = m.target_transform_
    assert verdict["name"] == "log" and verdict["accepted"] is True and verdict["head"] == "spline"
    assert {k: v for k, v in m.unary_forms_.items() if k in ("x1", "x2", "x3", "n0", "n1")} == {"x1": "sin", "x2": "log"}, m.unary_forms_
    Xte, yte = frame(1)

    def design(frame_):
        """The named forms of the inputs plus the linear one."""
        return np.column_stack([np.sin(frame_["x1"]), np.log(frame_["x2"]), frame_["x3"]])

    t = verdict["transform"]
    named = t.inverse(Ridge(1.0).fit(design(X), t.forward(y)).predict(design(Xte)))
    raw = Ridge(1.0).fit(X, y).predict(Xte)
    assert np.mean(np.abs(named - yte)) < 0.5 * np.mean(np.abs(raw - yte))
