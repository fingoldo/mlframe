"""``GatedOutlierEstimator`` -- classifier gate + regression blend for zero-inflated/point-mass targets.

Source idea (``1st_elo-merchant-category-recommendation.md``): many production targets are a mixture of a
degenerate point mass (a "no purchase"/"no claim"/exact-zero row) and a continuous distribution elsewhere
(spend amount, claim size, demand). Fitting one regressor across both regimes pulls its predictions toward
the point-mass value on the continuous rows and away from it on the point-mass rows -- the classic
zero-inflated-regression failure mode. This estimator instead trains a binary classifier for
"is this row the point mass" and a regressor on the non-point-mass rows only, then blends at predict time via
``p * point_mass_value + (1-p) * regressor.predict(X)`` where ``p`` is the classifier's predicted probability.

Distinct from :class:`mlframe.training.composite.estimator.CompositeTargetEstimator`, which reversibly
transforms a continuous target end-to-end (no discrete/continuous mixture, no classifier); and distinct from
``mlframe.training.composite._moe_gate``'s :class:`MoeSelectionGate`, which routes between precomputed expert
PREDICTIONS post-hoc rather than fitting the mixture itself. ``glm.py``'s Tweedie option handles zero-inflated
*positive-continuous* targets via a single GLM's compound-Poisson-Gamma likelihood -- a real alternative when a
Tweedie deviance is appropriate, but it doesn't generalize past that one distributional family the way an
explicit classifier + regressor split does (e.g. discrete point mass at any value, or a categorical secondary
model instead of a constant).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class GatedOutlierEstimator(BaseEstimator, RegressorMixin):
    """Classifier gate + regression blend for targets with a degenerate point mass plus a continuous regime.

    Parameters
    ----------
    regressor
        sklearn-compatible regressor prototype (``fit(X, y)`` / ``predict(X)``), cloned at fit time and
        fit ONLY on rows where the target is not the point-mass value.
    classifier
        sklearn-compatible probabilistic classifier prototype (``fit(X, y)`` / ``predict_proba(X)``), cloned
        at fit time and fit on the full training set against the binary point-mass indicator. Defaults to
        ``LogisticRegression(max_iter=1000)`` when None.
    point_mass_value
        The degenerate target value (e.g. ``0.0`` for "no purchase"). Rows with ``y == point_mass_value``
        (within ``point_mass_atol``) form the classifier's positive class and are excluded from the
        regressor's training rows.
    point_mass_atol
        Absolute tolerance for matching ``point_mass_value`` (handles float target noise around an exact
        value, e.g. ``0.0``).
    blend_value
        The value blended in for the point-mass probability mass at predict time. Defaults to
        ``point_mass_value`` itself (the natural zero-inflated case); pass a different constant, or a fitted
        secondary estimator's prediction via a custom subclass, for a "gate to a different regime" case.

    Attributes
    ----------
    classifier_, regressor_
        The fitted clones.
    point_mass_rate_
        Fraction of training rows that matched ``point_mass_value`` (diagnostic).
    """

    def __init__(
        self,
        regressor: Any,
        classifier: Optional[Any] = None,
        point_mass_value: float = 0.0,
        point_mass_atol: float = 1e-9,
        blend_value: Optional[float] = None,
    ) -> None:
        self.regressor = regressor
        self.classifier = classifier
        self.point_mass_value = point_mass_value
        self.point_mass_atol = point_mass_atol
        self.blend_value = blend_value

    def _is_point_mass(self, y: np.ndarray) -> np.ndarray:
        return np.isclose(y, self.point_mass_value, atol=self.point_mass_atol)

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> "GatedOutlierEstimator":
        y_arr = np.asarray(y, dtype=np.float64)
        is_point_mass = self._is_point_mass(y_arr)
        self.point_mass_rate_: float = float(is_point_mass.mean()) if y_arr.shape[0] else 0.0

        self.classifier_ = clone(self.classifier) if self.classifier is not None else LogisticRegression(max_iter=1000)
        if np.unique(is_point_mass).shape[0] < 2:
            # Degenerate training set (all point-mass or none): no meaningful gate to learn. Store the
            # constant class so predict_proba-shaped downstream logic still works via _constant_proba_.
            self._constant_proba_: Optional[float] = float(is_point_mass[0]) if is_point_mass.shape[0] else 0.0
        else:
            self._constant_proba_ = None
            fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            self.classifier_.fit(X, is_point_mass, **fit_kwargs)

        self.regressor_ = clone(self.regressor)
        non_point_mask = ~is_point_mass
        if non_point_mask.sum() == 0:
            # Every training row is the point mass -- nothing to fit a regressor on; predict() will lean
            # entirely on the classifier's (constant, ==1) probability and never call the regressor.
            self._regressor_fitted_ = False
        else:
            reg_X = X.loc[non_point_mask] if hasattr(X, "loc") else np.asarray(X)[non_point_mask]
            reg_y = y_arr[non_point_mask]
            reg_kwargs = {"sample_weight": sample_weight[non_point_mask]} if sample_weight is not None else {}
            self.regressor_.fit(reg_X, reg_y, **reg_kwargs)
            self._regressor_fitted_ = True

        return self

    def predict_proba_point_mass(self, X: Any) -> np.ndarray:
        """Return the classifier's predicted P(row is the point mass) as a 1-D array."""
        if self._constant_proba_ is not None:
            n = X.shape[0]
            return np.full(n, self._constant_proba_, dtype=np.float64)
        classes = list(self.classifier_.classes_)
        pos_idx = classes.index(True)
        return np.asarray(self.classifier_.predict_proba(X)[:, pos_idx], dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        p_point_mass = self.predict_proba_point_mass(X)
        blend_value = self.point_mass_value if self.blend_value is None else self.blend_value

        if not self._regressor_fitted_:
            return np.full(p_point_mass.shape[0], blend_value, dtype=np.float64)

        reg_pred = np.asarray(self.regressor_.predict(X), dtype=np.float64)
        return p_point_mass * blend_value + (1.0 - p_point_mass) * reg_pred


__all__ = ["GatedOutlierEstimator"]
