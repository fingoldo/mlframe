"""``recursive_multi_step_forecast``: fill short lags from the model's OWN previous-step predictions.

Source: 5th_m5-forecasting-accuracy.md -- "I also used the predicted value to predict the next (eg lag_7)."

mlframe's existing multi-step forecasters (``training._direct_horizon_bucket_forecaster.DirectHorizonBucketForecaster``,
``training.composite.direct_multi_horizon.DirectMultiHorizonEnsemble``) DELIBERATELY avoid this pattern --
both cite the classic recursive-forecasting failure mode (single-step prediction error compounds across
steps, since step 2's "ground truth" lag input is actually step 1's already-imperfect prediction). This
utility exists for the genuinely narrower case the source describes: when a short lag (e.g. ``lag_7``) is
NOT actually available at forecast time (true future ground truth doesn't exist yet, unlike a backtest where
it's tempting to use it), recursion is the only way to populate that feature at all -- not a preference, a
necessity. Use the DIRECT forecasters above whenever the lag CAN be avoided or replaced with an
origin-time-only feature; reach for this only when it genuinely can't.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def recursive_multi_step_forecast(
    model: Any,
    initial_features: pd.DataFrame,
    n_steps: int,
    lag_feature_name: str,
    update_features_fn: Callable[[pd.DataFrame, np.ndarray, int], pd.DataFrame],
) -> np.ndarray:
    """Forecast ``n_steps`` ahead, feeding each step's prediction into the next step's ``lag_feature_name``.

    Parameters
    ----------
    model
        Fitted model exposing ``predict(X) -> (n_rows,)``.
    initial_features
        ``(n_rows, n_features)`` feature frame for step 1 (must already contain any TRUE ground-truth lags
        available at forecast time -- only the RECURSIVE lag is filled by this loop).
    n_steps
        Number of forecast steps.
    lag_feature_name
        Column in ``initial_features`` that gets overwritten with the PREVIOUS step's prediction at every
        step after the first.
    update_features_fn
        ``update_features_fn(features, previous_predictions, step) -> updated_features`` -- caller-supplied
        hook for any OTHER per-step feature updates beyond the recursive lag itself (e.g. advancing a
        "days-since" counter); receives the frame with ``lag_feature_name`` already updated.

    Returns
    -------
    np.ndarray
        ``(n_steps, n_rows)`` predictions, one row per forecast step.
    """
    if lag_feature_name not in initial_features.columns:
        raise ValueError(f"recursive_multi_step_forecast: lag_feature_name {lag_feature_name!r} not in initial_features columns")

    features = initial_features.copy(deep=False)
    all_predictions: List[np.ndarray] = []

    for step in range(n_steps):
        pred = np.asarray(model.predict(features))
        all_predictions.append(pred)

        if step < n_steps - 1:
            features = features.copy(deep=False)
            features[lag_feature_name] = pred
            features = update_features_fn(features, pred, step)

    return np.stack(all_predictions, axis=0)


def diagnose_error_accumulation(
    model: Any,
    initial_features: pd.DataFrame,
    n_steps: int,
    lag_feature_name: str,
    update_features_fn: Callable[[pd.DataFrame, np.ndarray, int], pd.DataFrame],
    true_targets: np.ndarray,
    oracle_lag_values: Optional[np.ndarray] = None,
    accumulation_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Opt-in diagnostic: measure how much recursive-forecast error COMPOUNDS across the horizon.

    Runs ``recursive_multi_step_forecast`` on a validation window with KNOWN future targets, then reports
    the per-step error growth curve -- optionally against an "oracle" baseline that uses the TRUE (not
    self-predicted) lag at every step, i.e. the best-case, non-recursive error floor for the same model at
    that step. The gap between recursive error and the oracle floor isolates exactly how much of the error
    is compounding-induced rather than irreducible model noise, so a caller can pick a horizon beyond which
    to switch to a direct multi-output model instead of trusting the recursive feedback loop further.

    Parameters
    ----------
    true_targets
        ``(n_steps, n_rows)`` ground-truth values for each forecast step (only available in a backtest /
        validation window -- this diagnostic is for offline trustworthy-horizon calibration, not live use).
    oracle_lag_values
        ``(n_steps, n_rows)`` TRUE (ground-truth) value of ``lag_feature_name`` at each step, i.e. what a
        non-recursive direct-forecast baseline would see with perfect hindsight. When omitted, only the
        recursive error curve and its own growth ratio (relative to step 1) are reported.
    accumulation_threshold
        A step is flagged as no-longer-trustworthy once recursive MSE exceeds ``accumulation_threshold``
        times the oracle MSE at that step (or, with no oracle, ``accumulation_threshold`` times step-1's
        recursive MSE).

    Returns
    -------
    dict
        ``recursive_predictions`` ``(n_steps, n_rows)``, ``recursive_mse`` ``(n_steps,)``,
        ``growth_ratio`` ``(n_steps,)`` (recursive MSE relative to step 1), ``oracle_mse`` (``(n_steps,)``
        or ``None``), ``trustworthy_horizon`` (int -- number of leading steps, 1-indexed, that stay within
        ``accumulation_threshold``; equals ``n_steps`` if the threshold is never crossed).
    """
    if true_targets.shape != (n_steps, len(initial_features)):
        raise ValueError(f"diagnose_error_accumulation: true_targets shape {true_targets.shape} must be (n_steps={n_steps}, n_rows={len(initial_features)})")

    recursive_predictions = recursive_multi_step_forecast(model, initial_features, n_steps, lag_feature_name, update_features_fn)
    recursive_mse = np.mean((recursive_predictions - true_targets) ** 2, axis=1)
    growth_ratio = recursive_mse / recursive_mse[0]

    oracle_mse: Optional[np.ndarray] = None
    reference_mse = np.full(n_steps, recursive_mse[0])
    if oracle_lag_values is not None:
        if oracle_lag_values.shape != (n_steps, len(initial_features)):
            raise ValueError(f"diagnose_error_accumulation: oracle_lag_values shape {oracle_lag_values.shape} must be (n_steps={n_steps}, n_rows={len(initial_features)})")
        oracle_mse = np.empty(n_steps)
        for step in range(n_steps):
            oracle_features = initial_features.copy(deep=False)
            oracle_features[lag_feature_name] = oracle_lag_values[step]
            oracle_pred = np.asarray(model.predict(oracle_features))
            oracle_mse[step] = np.mean((oracle_pred - true_targets[step]) ** 2)
        reference_mse = oracle_mse

    trustworthy_horizon = n_steps
    for step in range(n_steps):
        if recursive_mse[step] > accumulation_threshold * reference_mse[step]:
            trustworthy_horizon = step
            break

    return {
        "recursive_predictions": recursive_predictions,
        "recursive_mse": recursive_mse,
        "growth_ratio": growth_ratio,
        "oracle_mse": oracle_mse,
        "trustworthy_horizon": trustworthy_horizon,
    }


__all__ = ["recursive_multi_step_forecast", "diagnose_error_accumulation"]
