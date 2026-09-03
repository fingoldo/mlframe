"""`CompositeTargetEstimator.fit` must reject inputs it cannot fit honestly, rather than fitting something else.

Both contracts here are pinned by sklearn's own estimator suite (`check_regressors_train`,
`check_supervised_y_no_nan`, `check_requires_y_none`) and both were unenforced: the existing length check ran
only on the `requires_base` path and compared y against the extracted base column, so a unary y-transform
accepted a mismatched y outright, and a non-finite y was never checked at all -- it flowed into the composite
transform, and the inner estimator trained on the result. Nothing raised; the failure surfaced as a nonsense
model much later, if at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlframe.training.composite import CompositeTargetEstimator


def _frame(n: int = 40) -> pd.DataFrame:
    """A minimal feature frame carrying the base column the `diff` transform needs."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f0": rng.normal(size=n), "__base__": rng.normal(size=n) + 10.0})


def _estimator(transform_name: str = "diff") -> CompositeTargetEstimator:
    """A composite estimator wired to a cheap inner regressor."""
    return CompositeTargetEstimator(base_estimator=Ridge(), transform_name=transform_name, base_column="__base__")


def test_mismatched_x_and_y_lengths_are_refused():
    """A y with fewer rows than X must raise, naming both counts."""
    X = _frame(40)
    y = np.arange(30, dtype=np.float64)  # 30 vs 40
    with pytest.raises(ValueError, match=r"40 rows but y has 30"):
        _estimator().fit(X, y)


def test_mismatched_lengths_are_refused_for_a_unary_transform_too():
    """The unary (`requires_base=False`) path had no length check at all, since it never touches a base column."""
    X = _frame(40)
    y = np.abs(np.arange(30, dtype=np.float64)) + 1.0
    with pytest.raises(ValueError, match=r"40 rows but y has 30"):
        _estimator(transform_name="log_y").fit(X, y)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_y_is_refused(bad):
    """NaN and both infinities are rejected, and the message says how many rows are affected."""
    X = _frame(40)
    y = np.linspace(1.0, 2.0, 40)
    y[7] = bad
    with pytest.raises(ValueError, match=r"y contains 1 non-finite value"):
        _estimator().fit(X, y)


def test_y_none_names_the_broken_contract():
    """A supervised estimator handed `y=None` must say so, not surface an internal coercion helper's message."""
    with pytest.raises(ValueError, match=r"requires y to be passed, but the target y is None"):
        _estimator().fit(_frame(40), None)


def test_a_well_formed_fit_still_succeeds():
    """The guards must not reject a legitimate call -- without this the four above could pass by fitting nothing."""
    X = _frame(60)
    y = X["__base__"].to_numpy() + np.linspace(0.0, 1.0, 60)
    est = _estimator().fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (60,)
    assert np.isfinite(preds).all(), "a clean fit must produce finite predictions"
