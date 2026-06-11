"""Regression tests for the E8 *hasattr-trap* in ``CompositeTargetEstimator``'s
delegated inner-model attribute accessors.

E8 (future audit item, ``audit/composite_audit_2026_06_10/future_items.json``):
``feature_importances_`` / ``coef_`` / ``intercept_`` / ``booster_`` used
``getattr(inner, attr, None)``. Because the wrapper defines the property
descriptor on the class, the property ALWAYS executed post-fit and returned
``None`` whenever the FITTED inner genuinely lacked the attribute. That made
``hasattr(wrapper, attr)`` report ``True`` with a ``None`` payload -- the
inverse of sklearn semantics. Downstream consumers that gate on
``hasattr(est, "feature_importances_")`` (``SelectFromModel``, SHAP, plotting
helpers) then read a silent ``None`` and crash with an opaque ``TypeError``.

Post-fix contract (mirrors a real sklearn estimator):
- pre-fit  -> ``NotFittedError`` (unchanged)
- fitted, inner HAS attr     -> the real value; ``hasattr`` is ``True``
- fitted, inner LACKS attr   -> ``AttributeError`` propagates; ``hasattr`` is
                                ``False`` (so ``getattr(est, attr, default)``
                                returns ``default``, the consumer-safe path)

These tests FAIL on the pre-fix ``getattr(..., None)`` bodies: the
``hasattr(...) is False`` assertions trip because the old code returned ``None``
(making ``hasattr`` ``True``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


def _fit_linreg_wrapper(n: int = 60) -> CompositeTargetEstimator:
    rng = np.random.default_rng(0)
    base = rng.uniform(1.0, 5.0, size=n)
    X = pd.DataFrame({"base": base, "f1": rng.normal(size=n)})
    y = 2.0 * base + 0.5 * X["f1"].to_numpy() + rng.normal(scale=0.1, size=n)
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="diff", base_column="base"
    )
    est.fit(X, y)
    return est


def _fit_rf_wrapper(n: int = 80) -> CompositeTargetEstimator:
    rng = np.random.default_rng(1)
    base = rng.uniform(1.0, 5.0, size=n)
    X = pd.DataFrame({"base": base, "f1": rng.normal(size=n)})
    y = 2.0 * base + rng.normal(scale=0.2, size=n)
    est = CompositeTargetEstimator(
        base_estimator=RandomForestRegressor(n_estimators=5, random_state=0),
        transform_name="diff",
        base_column="base",
    )
    est.fit(X, y)
    return est


# ---------------------------------------------------------------------------
# Core hasattr-trap regression: linear inner lacks tree/booster attrs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["feature_importances_", "booster_"])
def test_linreg_inner_missing_attr_is_not_advertised(attr):
    """A fitted ``LinearRegression`` inner has no ``feature_importances_`` /
    ``booster_``; the wrapper MUST report ``hasattr == False`` (was ``True``
    with a ``None`` value -- the trap)."""
    est = _fit_linreg_wrapper()
    assert hasattr(est, attr) is False, (
        f"hasattr(wrapper, {attr!r}) must be False when the fitted inner lacks "
        f"it (pre-fix returned None, making hasattr True -- the trap)"
    )


@pytest.mark.parametrize("attr", ["feature_importances_", "booster_"])
def test_linreg_inner_missing_attr_raises_attributeerror(attr):
    """Direct access raises ``AttributeError`` (NOT a silent ``None``)."""
    est = _fit_linreg_wrapper()
    with pytest.raises(AttributeError, match=attr):
        getattr(est, attr)


@pytest.mark.parametrize("attr", ["feature_importances_", "booster_"])
def test_getattr_with_default_returns_default_not_none_value(attr):
    """The consumer-safe ``getattr(est, attr, sentinel)`` path now returns the
    sentinel (because the property raises ``AttributeError``); pre-fix it
    returned ``None`` regardless of the sentinel, silently corrupting the
    fallback semantics that ``SelectFromModel`` / SHAP rely on."""
    est = _fit_linreg_wrapper()
    sentinel = object()
    assert getattr(est, attr, sentinel) is sentinel


# ---------------------------------------------------------------------------
# Positive path: attrs that DO exist on the inner stay exposed with values.
# ---------------------------------------------------------------------------

def test_linreg_inner_present_attrs_exposed():
    """``coef_`` / ``intercept_`` exist on a fitted ``LinearRegression`` and
    must be advertised AND non-None (no over-correction)."""
    est = _fit_linreg_wrapper()
    assert hasattr(est, "coef_") is True
    assert est.coef_ is not None
    assert isinstance(est.coef_, np.ndarray)
    assert hasattr(est, "intercept_") is True
    assert est.intercept_ is not None
    # LinearRegression has no tree importances / booster.
    assert hasattr(est, "coef_") and not hasattr(est, "feature_importances_")


def test_tree_inner_feature_importances_exposed():
    """A fitted ``RandomForestRegressor`` inner DOES expose
    ``feature_importances_``; the wrapper must surface it (not hide it)."""
    est = _fit_rf_wrapper()
    assert hasattr(est, "feature_importances_") is True
    fi = est.feature_importances_
    assert isinstance(fi, np.ndarray)
    # 2 input columns (base + f1); group_column not dropped on a non-grouped transform.
    assert fi.shape == (2,)
    # RandomForest has no linear coef_.
    assert hasattr(est, "coef_") is False


# ---------------------------------------------------------------------------
# Pre-fit contract preserved: NotFittedError still wins for every accessor.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["feature_importances_", "coef_", "intercept_", "booster_"])
def test_prefit_access_raises_notfittederror(attr):
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="diff", base_column="base"
    )
    with pytest.raises(NotFittedError):
        getattr(est, attr)
