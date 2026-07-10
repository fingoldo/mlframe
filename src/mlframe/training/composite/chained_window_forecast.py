"""``ChainedWindowForecaster``: rolling self-referential forecast chaining across time windows.

Source: Optiver Realized Volatility Prediction 3rd place's "300 seconds model" -- slice the observation
window in half, train a stage-1 model to predict window[t]'s target from window[t-1]'s features, then APPLY
that fitted stage-1 model to window[t]'s own (fully observed) features -- extrapolating the learned
window-to-window mapping one step further -- and feed that extrapolated value as an extra input feature into
a stage-2 model that predicts the TRUE target of the unobserved next window, alongside window[t]'s directly
observed features. Chaining lets the pipeline exploit real, observed target-window features that a naive
single model trained only on window[t-1] wouldn't otherwise leverage for the window-t+1 forecast.

No OOF plumbing is needed here (unlike :class:`MultiStageMetaFeatureStacker`'s auxiliary-target stacking):
stage 1's proxy target (window[t]'s own computed quantity) and stage 2's true target (window[t+1]'s target)
are DIFFERENT targets for DIFFERENT time windows, so stage 1 is simply fit once on ALL rows (``X_prev``,
``y_curr``) and then evaluated on ``X_curr`` -- a legitimate one-step-ahead extrapolation, not an in-sample
prediction of anything stage 1 was ever fit to predict.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)


class ChainedWindowForecaster(BaseEstimator, RegressorMixin):
    """Stage 1 (window[t-1] -> window[t] proxy target) extrapolated onto window[t]'s own features, feeding
    stage 2 (window[t]'s features + the extrapolated value -> window[t+1]'s true target).

    Parameters
    ----------
    stage1_estimator
        sklearn-compatible estimator prototype, cloned at fit time. Fit on ``(X_prev, y_curr)`` -- the
        earlier window's features predicting the later window's proxy target.
    stage2_estimator
        sklearn-compatible estimator prototype, cloned at fit time. Fit on ``X_curr`` plus the stage-1
        extrapolated feature, predicting ``y_target`` -- the TRUE (later, currently-unobserved-at-inference)
        target.
    chained_feature_name
        Column name for the injected stage-1 extrapolated feature.

    Attributes
    ----------
    stage1_model_, stage2_model_
        The fitted clones.
    """

    def __init__(self, stage1_estimator: Any, stage2_estimator: Any, chained_feature_name: str = "stage1_chained_pred") -> None:
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.chained_feature_name = chained_feature_name

    def fit(
        self,
        X_prev: Any,
        X_curr: Any,
        y_curr: np.ndarray,
        y_target: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ChainedWindowForecaster":
        """
        Parameters
        ----------
        X_prev
            Earlier window's features, ``(n, d)``.
        X_curr
            Later (but still fully observed) window's features, ``(n, d)`` -- same feature space as
            ``X_prev``, same rows (one row per forecast instance, its two consecutive windows).
        y_curr
            Proxy target computed FROM window ``X_curr`` (e.g. that window's own realized volatility) --
            what stage 1 learns to predict from the PRECEDING window's features.
        y_target
            The TRUE target for the window AFTER ``X_curr`` -- unobserved at inference time, what stage 2
            ultimately forecasts.
        """
        self.stage1_model_ = clone(self.stage1_estimator)
        self.stage1_model_.fit(X_prev, np.asarray(y_curr, dtype=np.float64))

        chained_pred = np.asarray(self.stage1_model_.predict(X_curr), dtype=np.float64)
        X2 = self._concat_chained(X_curr, chained_pred)

        self.stage2_model_ = clone(self.stage2_estimator)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.stage2_model_.fit(X2, np.asarray(y_target, dtype=np.float64), **fit_kwargs)
        return self

    def _concat_chained(self, X_curr: Any, chained_pred: np.ndarray) -> Any:
        if isinstance(X_curr, pd.DataFrame):
            out = X_curr.copy()
            out[self.chained_feature_name] = chained_pred
            return out
        try:
            import polars as pl
            if isinstance(X_curr, pl.DataFrame):
                return X_curr.with_columns(pl.Series(self.chained_feature_name, chained_pred))
        except ImportError:
            pass
        X_arr = np.asarray(X_curr, dtype=np.float64)
        return np.concatenate([X_arr, chained_pred.reshape(-1, 1)], axis=1)

    def predict(self, X_curr: Any) -> np.ndarray:
        """Forecast the next window's target from ``X_curr`` (the latest fully-observed window's features)."""
        chained_pred = np.asarray(self.stage1_model_.predict(X_curr), dtype=np.float64)
        X2 = self._concat_chained(X_curr, chained_pred)
        return np.asarray(self.stage2_model_.predict(X2))


__all__ = ["ChainedWindowForecaster"]
