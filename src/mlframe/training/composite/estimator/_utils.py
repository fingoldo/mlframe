"""Property / accessor / shim methods for ``CompositeTargetEstimator``.

Carved out of ``_composite_target_estimator.py`` to keep the parent below the 1k-line monolith threshold. The functions defined here become bound methods on ``CompositeTargetEstimator`` via direct class-attribute assignment at the parent's bottom (mirror of the ``RFECV.fit`` carve pattern). Behavioural identity is preserved bit-for-bit; downstream ``isinstance`` / ``hasattr`` checks see the same class object.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.exceptions import NotFittedError


def _require_fitted(self, attr: str) -> Any:
    """sklearn convention: pre-fit access to ``feature_importances_``/``coef_``/
    ``intercept_`` raises ``NotFittedError`` (was: silent ``None`` return).
    """
    est = getattr(self, "estimator_", None)
    if est is None:
        raise NotFittedError(f"CompositeTargetEstimator has no fitted ``estimator_``; " f"call .fit(...) before accessing .{attr}.")
    return est


def _require_inner_attr(self, attr: str) -> Any:
    """Resolve a delegated inner-model attribute, mirroring sklearn semantics.

    E8 fix (the *hasattr-trap*): the previous bodies used
    ``getattr(inner, attr, None)``. After ``fit`` the descriptor on the
    wrapper class always exists, so the property executed and returned
    ``None`` whenever the FITTED inner genuinely lacked the attribute
    (e.g. a ``LinearRegression`` inner has no ``feature_importances_`` /
    ``booster_``). That made ``hasattr(wrapper, attr)`` report ``True``
    with a ``None`` payload -- the exact opposite of what every sklearn
    consumer expects. ``SelectFromModel`` / SHAP / plotting helpers gate
    on ``hasattr(est, "feature_importances_")`` and then *use* the value;
    a silent ``None`` slips past the guard and crashes downstream with an
    opaque ``TypeError``.

    A real sklearn estimator that lacks ``feature_importances_`` has NO
    such attribute, so ``hasattr`` is ``False``. We restore that contract
    by delegating with a **no-default** ``getattr``: pre-fit ->
    ``NotFittedError`` (via ``_require_fitted``); fitted-but-missing ->
    ``AttributeError`` (propagated, so ``hasattr`` is ``False``); fitted-
    and-present -> the real value. The wrapper now advertises exactly the
    inner-model capabilities that actually exist.
    """
    est = self._require_fitted(attr)
    try:
        return getattr(est, attr)
    except AttributeError as exc:
        # Re-raise with the wrapper context but keep ``AttributeError`` so
        # ``hasattr`` short-circuits to ``False`` (sklearn convention).
        raise AttributeError(
            f"CompositeTargetEstimator: inner estimator "
            f"{type(est).__name__!r} has no attribute {attr!r}; the wrapper "
            f"only exposes inner-model attributes that the fitted inner "
            f"actually provides."
        ) from exc


def feature_importances_(self) -> np.ndarray:
    """Delegate to the fitted inner estimator's ``feature_importances_`` (bound as a property on the wrapper).

    Raises ``NotFittedError`` before ``fit`` and ``AttributeError`` if the fitted inner has none.
    """
    return np.asarray(self._require_inner_attr("feature_importances_"))


def coef_(self) -> np.ndarray:
    """Delegate to the fitted inner estimator's ``coef_`` (bound as a property on the wrapper).

    Raises ``NotFittedError`` before ``fit`` and ``AttributeError`` if the fitted inner has none.
    """
    return np.asarray(self._require_inner_attr("coef_"))


def intercept_(self) -> float:
    """Delegate to the fitted inner estimator's ``intercept_`` (bound as a property on the wrapper).

    Raises ``NotFittedError`` before ``fit`` and ``AttributeError`` if the fitted inner has none.
    """
    return self._require_inner_attr("intercept_")


def get_booster(self):
    """XGBoost shim. Uses the same ``_require_fitted`` path as the
    sklearn-convention properties so the error class is
    ``NotFittedError`` rather than the bespoke ``RuntimeError`` it
    used to raise - callers can catch both shapes uniformly.

    E8: a fitted inner without ``get_booster`` raises ``AttributeError``
    (propagated by ``_require_inner_attr`` would not apply here because
    ``get_booster`` is a method, not a value -- but the same no-default
    semantics hold: ``getattr`` with no default lets the missing-method
    ``AttributeError`` surface so ``hasattr(wrapper, "get_booster")`` is
    ``True`` only when the inner truly has it)."""
    return self._require_inner_attr("get_booster")()


def booster_(self):
    """LightGBM shim. Raises ``NotFittedError`` when called before ``fit``
    so callers fail loudly instead of receiving a silent ``None`` and
    crashing downstream (sklearn convention).

    E8: a fitted non-LightGBM inner (no ``booster_``) now raises
    ``AttributeError`` instead of returning ``None``, so
    ``hasattr(wrapper, "booster_")`` correctly reports ``False`` and the
    LightGBM-specific code paths that gate on it are not mis-routed."""
    return self._require_inner_attr("booster_")


def n_features_in_(self) -> int | None:
    """Pre-fit return ``None`` (NOT raise) so introspection tools that
    defensively check ``if est.n_features_in_ is None: ...`` keep working
    across the wrapper. Distinct from the ``feature_importances_`` /
    ``coef_`` / ``intercept_`` properties above which DO raise
    ``NotFittedError`` pre-fit: those are coefficient-style values that
    callers expect to be present once fitted, and silently returning
    ``None`` for them is a footgun. ``n_features_in_`` is a metadata
    scalar with a long-standing None-pre-fit convention in mlframe.

    E11 fix: for GROUPED transforms the wrapper's ``feature_names_in_``
    records the columns it SAW (including ``group_column``), but the inner
    estimator is fit AFTER ``group_column`` is dropped, so a bare delegation
    to ``inner.n_features_in_`` reports one fewer feature than
    ``len(feature_names_in_)`` -- violating the sklearn invariant
    ``n_features_in_ == len(feature_names_in_)``. The wrapper therefore
    stamps the count it saw at the X boundary (``self._n_features_in_wrapper``)
    in both the ``fit`` and ``from_fitted_inner`` routes and that value is
    preferred here. A class-level ``property`` shadows any instance attribute
    of the same name, so the stored value lives under a distinct private name
    rather than being assignable as ``self.n_features_in_``.
    """
    wrapper_n = getattr(self, "_n_features_in_wrapper", None)
    if wrapper_n is not None:
        return int(wrapper_n)
    est = getattr(self, "estimator_", None)
    if est is None:
        return None
    return getattr(est, "n_features_in_", None)
