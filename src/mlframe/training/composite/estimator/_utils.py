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
        raise NotFittedError(
            f"CompositeTargetEstimator has no fitted ``estimator_``; "
            f"call .fit(...) before accessing .{attr}."
        )
    return est


def feature_importances_(self) -> np.ndarray | None:
    return getattr(self._require_fitted("feature_importances_"), "feature_importances_", None)


def coef_(self) -> np.ndarray | None:
    return getattr(self._require_fitted("coef_"), "coef_", None)


def intercept_(self) -> float | None:
    return getattr(self._require_fitted("intercept_"), "intercept_", None)


def get_booster(self):
    """XGBoost shim. Uses the same ``_require_fitted`` path as the
    sklearn-convention properties so the error class is
    ``NotFittedError`` rather than the bespoke ``RuntimeError`` it
    used to raise - callers can catch both shapes uniformly."""
    return self._require_fitted("get_booster").get_booster()


def booster_(self):
    """LightGBM shim. Raises ``NotFittedError`` when called before
    ``fit`` so callers fail loudly instead of receiving a silent
    ``None`` and crashing downstream (sklearn convention)."""
    return getattr(self._require_fitted("booster_"), "booster_", None)


def n_features_in_(self) -> int | None:
    """Pre-fit return ``None`` (NOT raise) so introspection tools that
    defensively check ``if est.n_features_in_ is None: ...`` keep working
    across the wrapper. Distinct from the ``feature_importances_`` /
    ``coef_`` / ``intercept_`` properties above which DO raise
    ``NotFittedError`` pre-fit: those are coefficient-style values that
    callers expect to be present once fitted, and silently returning
    ``None`` for them is a footgun. ``n_features_in_`` is a metadata
    scalar with a long-standing None-pre-fit convention in mlframe.
    """
    est = getattr(self, "estimator_", None)
    if est is None:
        return None
    return getattr(est, "n_features_in_", None)
