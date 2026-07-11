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

from typing import Any, Callable, Dict, List

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


__all__ = ["recursive_multi_step_forecast"]
