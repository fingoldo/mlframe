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

Transductive stage-1 pretraining (``X_prev_extra``/``y_curr_extra`` in :meth:`fit`): stage 1's proxy target is
computed FROM window[t] -- which is fully observed in the TEST set too (window[t] precedes the true
unobserved forecast window[t+1], and is available at inference time, not just at train time). There is no
label leakage in training stage 1 on the test set's own early-window data alongside train's: stage 1 never
sees ``y_target`` (test's real, unobserved label), only ``(X_prev, y_curr)`` pairs computable from ANY row,
labeled or not. Folding test's own ``(X_prev, y_curr)`` rows into the stage-1 fit gives it more data to learn
the window-to-window mapping from, directly the source technique's "trained ... using both train and test
data" detail.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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
        X_prev_extra: Optional[Any] = None,
        y_curr_extra: Optional[np.ndarray] = None,
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
        X_prev_extra, y_curr_extra
            Optional ADDITIONAL ``(X_prev, y_curr)``-shaped rows folded into the STAGE-1 fit only (never
            stage 2, which strictly needs real ``y_target`` labels) -- e.g. the test set's own early-window
            rows, whose proxy target is fully observed despite the test set having no real label yet (see
            the module docstring's "Transductive stage-1 pretraining" note). Stacked onto ``X_prev``/``y_curr``
            before the stage-1 fit; stage 2 is unaffected.
        """
        if X_prev_extra is not None:
            stage1_X = self._stack_rows(X_prev, X_prev_extra)
            stage1_y = np.concatenate([np.asarray(y_curr, dtype=np.float64), np.asarray(y_curr_extra, dtype=np.float64)])
        else:
            stage1_X = X_prev
            stage1_y = np.asarray(y_curr, dtype=np.float64)

        self.stage1_model_ = clone(self.stage1_estimator)
        self.stage1_model_.fit(stage1_X, stage1_y)

        chained_pred = np.asarray(self.stage1_model_.predict(X_curr), dtype=np.float64)
        X2 = self._concat_chained(X_curr, chained_pred)

        self.stage2_model_ = clone(self.stage2_estimator)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.stage2_model_.fit(X2, np.asarray(y_target, dtype=np.float64), **fit_kwargs)
        return self

    @staticmethod
    def _stack_rows(X: Any, X_extra: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return pd.concat([X.reset_index(drop=True), pd.DataFrame(X_extra, columns=X.columns).reset_index(drop=True) if not isinstance(X_extra, pd.DataFrame) else X_extra.reset_index(drop=True)], axis=0, ignore_index=True)
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                X_extra_pl = X_extra if isinstance(X_extra, pl.DataFrame) else pl.DataFrame(X_extra, schema=X.columns)
                return pl.concat([X, X_extra_pl])
        except ImportError:
            pass
        return np.concatenate([np.asarray(X, dtype=np.float64), np.asarray(X_extra, dtype=np.float64)], axis=0)

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

    def diagnose_error_accumulation(
        self,
        X_curr_sequence: List[Any],
        y_target_sequence: List[np.ndarray],
        accumulation_threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """Opt-in diagnostic: measure how STAGE-1 EXTRAPOLATION DRIFT compounds as this already-fitted
        forecaster is reused, without refitting, across a SEQUENCE of successive future windows.

        ``stage1_model_`` is fit once on ``(X_prev, y_curr)`` pairs from the fit-time window; even a single
        :meth:`predict` call already extrapolates it onto ``X_curr`` (see the module docstring). Reusing the
        same fitted forecaster across many later windows without refitting means each window's stage-1 call
        can drift further from the distribution stage1 was trained on, and stage2 blindly trusts whatever
        stage1 hands it as the chained feature -- so stage-1 drift propagates straight into the final
        forecast. This isolates how far into such a chain of windows the forecaster stays trustworthy,
        mirroring ``inference.recursive_forecast.diagnose_error_accumulation``'s per-step growth-ratio /
        trustworthy-horizon reporting, but for THIS class's window-chain (not lag-feedback) failure mode.

        Parameters
        ----------
        X_curr_sequence
            Ordered list of ``X_curr``-shaped window feature frames/arrays, one per successive chain
            position (index 0 assumed the window nearest the one the forecaster was fit on).
        y_target_sequence
            Matching list of ground-truth targets for each window -- only available in a backtest /
            validation setting, like the sibling diagnostic this is for offline calibration, not live use.
        accumulation_threshold
            A chain position is flagged as no-longer-trustworthy once its MSE exceeds
            ``accumulation_threshold`` times position 0's MSE.

        Returns
        -------
        dict
            ``chain_predictions`` (list of ``(n_rows,)`` arrays, one per position), ``chain_mse``
            ``(n_positions,)``, ``growth_ratio`` ``(n_positions,)`` (MSE relative to position 0),
            ``trustworthy_horizon`` (int -- number of leading chain positions, 1-indexed, that stay within
            ``accumulation_threshold``; equals ``len(X_curr_sequence)`` if the threshold is never crossed).
        """
        if len(X_curr_sequence) != len(y_target_sequence):
            raise ValueError(f"diagnose_error_accumulation: X_curr_sequence has {len(X_curr_sequence)} windows but y_target_sequence has {len(y_target_sequence)}")
        if len(X_curr_sequence) == 0:
            raise ValueError("diagnose_error_accumulation: X_curr_sequence must have at least one window")

        n_positions = len(X_curr_sequence)
        chain_predictions: List[np.ndarray] = []
        chain_mse = np.empty(n_positions)
        for i in range(n_positions):
            pred = self.predict(X_curr_sequence[i])
            chain_predictions.append(pred)
            chain_mse[i] = np.mean((pred - np.asarray(y_target_sequence[i], dtype=np.float64)) ** 2)

        growth_ratio = chain_mse / chain_mse[0]

        trustworthy_horizon = n_positions
        for i in range(n_positions):
            if chain_mse[i] > accumulation_threshold * chain_mse[0]:
                trustworthy_horizon = i
                break

        return {
            "chain_predictions": chain_predictions,
            "chain_mse": chain_mse,
            "growth_ratio": growth_ratio,
            "trustworthy_horizon": trustworthy_horizon,
        }


__all__ = ["ChainedWindowForecaster"]
