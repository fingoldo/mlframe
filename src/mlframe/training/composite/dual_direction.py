"""``DualDirectionCompositeEstimator`` -- shape/scale two-stage composite (LANL earthquake 1st-place pattern).

Source: 1st_lanl-earthquake-prediction.md -- "normalized the ttf targets to be in range 0-1 and then predicts
this normalized target and scales it by ttf+tsl prediction": a SHAPE model predicts a bounded, cycle-normalized
value; a separate SCALE model predicts the magnitude/duration multiplier; the two combine as
``y = shape_prediction * scale_prediction``.

mlframe's existing ``CompositeTargetEstimator`` ratio transform (``T = y / base``, inverted as
``y = T_pred * base``) already implements the shape*scale MATH -- but ``base_column`` is always a raw,
already-materialized column in ``X`` (a plain lookup, never a model call), so nothing automates the two-stage
pipeline the source idea actually describes: fit a SCALE model, and feed its PREDICTION (not a precomputed
feature) as the ratio's base at both fit and predict time. This class is that thin auto-wiring wrapper --
no new transform math, just orchestration:

1. ``fit``: leakage-free OOF scale predictions (``sklearn.model_selection.cross_val_predict`` -- reusing an
   existing, well-tested library primitive rather than writing a new OOF loop) become the ratio's base column
   for fitting the shape-side ``CompositeTargetEstimator``; the scale estimator is then refit on the FULL
   training data for predict-time use (OOF-then-refit-on-full is the standard mlframe pattern used elsewhere
   for this same reason -- OOF predictions avoid the scale model leaking its own train-fit bias into the shape
   model's training signal, while the refit-on-full model gives the best available scale estimator for new data).
2. ``predict``: the refit scale model predicts on new ``X``; its prediction becomes the ratio base column fed
   to the fitted shape-side composite estimator.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict

from mlframe.metrics.core import fast_r2_score
from mlframe.training.composite.estimator import CompositeTargetEstimator

logger = logging.getLogger(__name__)

_SCALE_PRED_COLUMN = "__dual_direction_scale_pred__"


class DualDirectionCompositeEstimator(BaseEstimator, RegressorMixin):
    """Two-stage shape*scale composite: ``y_pred = shape_model_ratio_pred * scale_model_pred``.

    Parameters
    ----------
    scale_estimator
        Sklearn-compatible regressor predicting the SCALE/magnitude target (e.g. time-since + time-to sum).
        Cloned internally; the unfitted prototype passed in stays clean.
    shape_estimator
        Sklearn-compatible regressor for the SHAPE-normalized target, wrapped internally in a
        ``CompositeTargetEstimator(transform_name="ratio", base_column=<scale prediction column>)``.
    n_splits
        K-fold count for the leakage-free OOF scale predictions used at shape-model fit time.
    fallback_predict
        Forwarded to the inner ``CompositeTargetEstimator`` (``"y_train_median"`` or ``"nan"`` for rows where
        the scale prediction is ``<= 0``/``inf``, i.e. the ratio's domain is violated).
    random_state
        Forwarded to the ``cross_val_predict`` KFold splitter for the scale OOF predictions.

    Attributes (post-``fit``)
    --------------------------
    oof_scale_predictions_
        The leakage-free ``cross_val_predict`` scale predictions computed during ``fit``, aligned to
        ``X_train``'s original row order -- otherwise computed internally and discarded. Lets a caller
        diagnose the scale stage (e.g. correlate against a known ground-truth scale) without re-running
        ``cross_val_predict`` themselves.
    oof_scale_score_
        ``r2_score(scale_y, oof_scale_predictions_)`` -- an honest (leakage-free) quality signal for the
        scale stage alone. Low/negative values flag a scale model too weak to support the shape*scale
        decomposition even before looking at end-to-end ``y`` error.
    shape_transform_target_
        The ratio-transformed shape target actually fed to the shape estimator, ``y_train / oof_scale_predictions_``
        (``NaN`` where ``oof_scale_predictions_ <= 0``, mirroring the ``"ratio"`` transform's domain guard).
        Useful for inspecting the shape target's distribution for domain-shift / heavy-tail issues.
    """

    oof_scale_predictions_: np.ndarray
    oof_scale_score_: float
    shape_transform_target_: np.ndarray

    def __init__(
        self,
        scale_estimator: Any,
        shape_estimator: Any,
        n_splits: int = 5,
        fallback_predict: str = "y_train_median",
        random_state: int = 0,
    ) -> None:
        self.scale_estimator = scale_estimator
        self.shape_estimator = shape_estimator
        self.n_splits = n_splits
        self.fallback_predict = fallback_predict
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any, scale_y: Any) -> "DualDirectionCompositeEstimator":
        """Fit the two-stage estimator.

        Parameters
        ----------
        X
            Feature frame.
        y
            The TRUE-units target (``shape * scale``).
        scale_y
            The scale/magnitude auxiliary target (e.g. time-since + time-to sum in the source idea).
        """
        from sklearn.model_selection import KFold

        y_arr = np.asarray(y, dtype=np.float64)
        scale_y_arr = np.asarray(scale_y, dtype=np.float64)
        if len(y_arr) != len(X) or len(scale_y_arr) != len(X):
            raise ValueError(f"DualDirectionCompositeEstimator.fit: X has {len(X)} rows but y has {len(y_arr)} and scale_y has {len(scale_y_arr)} -- misaligned inputs.")

        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_scale_pred = np.asarray(cross_val_predict(clone(self.scale_estimator), X, scale_y_arr, cv=cv), dtype=np.float64)

        # Public diagnostic attributes: these are already computed as part of the OOF-then-refit-on-full
        # pipeline above -- exposing them costs O(n) (a division + an r2_score call), not a second CV pass.
        self.oof_scale_predictions_ = oof_scale_pred
        self.oof_scale_score_ = float(fast_r2_score(scale_y_arr, oof_scale_pred))
        with np.errstate(divide="ignore", invalid="ignore"):
            shape_transform_target = y_arr / oof_scale_pred
        shape_transform_target[oof_scale_pred <= 0] = np.nan
        self.shape_transform_target_ = shape_transform_target

        X_with_scale = X.copy(deep=False)
        X_with_scale[_SCALE_PRED_COLUMN] = oof_scale_pred

        self.shape_estimator_ = CompositeTargetEstimator(
            base_estimator=clone(self.shape_estimator),
            transform_name="ratio",
            base_column=_SCALE_PRED_COLUMN,
            fallback_predict=self.fallback_predict,
        )
        self.shape_estimator_.fit(X_with_scale, y_arr)

        # Refit the scale estimator on the FULL training data for predict-time use -- the OOF predictions above
        # exist only to keep the shape model's training signal leakage-free; a full-data refit is the best
        # available scale estimator once that purpose is served (standard OOF-then-refit-on-full pattern).
        self.scale_estimator_ = clone(self.scale_estimator)
        self.scale_estimator_.fit(X, scale_y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ``y = shape_prediction * scale_prediction`` on new rows."""
        self._require_fitted()
        scale_pred = np.asarray(self.scale_estimator_.predict(X), dtype=np.float64)
        X_with_scale = X.copy(deep=False)
        X_with_scale[_SCALE_PRED_COLUMN] = scale_pred
        return np.asarray(self.shape_estimator_.predict(X_with_scale))

    def predict_scale(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the scale/magnitude component alone (diagnostic access to stage 1)."""
        self._require_fitted()
        return np.asarray(self.scale_estimator_.predict(X))

    def _require_fitted(self) -> None:
        if not hasattr(self, "shape_estimator_") or not hasattr(self, "scale_estimator_"):
            raise ValueError("DualDirectionCompositeEstimator: call fit(X, y, scale_y) before predict.")


__all__ = ["DualDirectionCompositeEstimator"]
