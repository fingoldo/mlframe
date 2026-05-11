"""OPEN-10 helpers: composite_predictions_as_feature attaches a fitted wrapper's predictions to a dataframe as a new column; composite_oof_predictions runs K-fold OOF stacking to avoid in-sample leakage. Both use ``CompositeTargetEstimator``-style wrappers passed in by the caller; no compile-time dep on the wrapper class itself."""


from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# composite_predictions_as_feature (R10c brainstorm #10; composite x FE-pipeline stacking).
#
# Exposes a fitted composite-target model's predictions as an engineered feature column on the input dataframe. The downstream FE pipeline / final model treats it like any other feature, enabling a single-final-model alternative to the cross-target NNLS ensemble: the final model corrects the composite's bias regions on the SAME row using all other features.
#
# Two variants ship together:
# - ``composite_predictions_as_feature(wrapper, df, column_name=None)`` -- attach the wrapper's predict() output as a new column. Useful at INFERENCE time when the wrapper is already fitted (and at train time when the wrapper was fitted on a SEPARATE holdout so the in-sample predictions aren't optimistic).
# - ``composite_oof_predictions(wrapper_factory, X, y, n_splits=5, random_state=42)`` -- OOF stacking: K-fold CV where each row's prediction comes from a wrapper trained on the OTHER folds. Avoids the in-sample-optimism trap. Caller passes a zero-arg ``wrapper_factory`` that builds a fresh (unfitted) wrapper each call.
# ----------------------------------------------------------------------


def composite_predictions_as_feature(
    wrapper: Any,
    df: Any,
    *,
    column_name: Optional[str] = None,
    fallback_value: Optional[float] = None,
) -> Any:
    """Attach a fitted composite wrapper's predictions to ``df`` as a new column.

    Parameters
    ----------
    wrapper
        Fitted ``CompositeTargetEstimator`` (or any predict-supporting object). Caller is responsible for ``fit``-ing it on data that does NOT include ``df``'s rows (otherwise the column carries in-sample optimism that the downstream final model trains on -- effectively leakage).
    df
        pandas / polars frame. The wrapper's ``predict(df)`` is called; the resulting 1-D array is attached as a new column.
    column_name
        Override for the new column name. Default: ``"composite_pred__{transform_name}__{base_column}"`` derived from the wrapper's attributes, with a generic ``"composite_pred"`` fallback when the wrapper doesn't expose them.
    fallback_value
        When the wrapper's ``predict`` raises (e.g. missing base column), fill the new column with this value instead of propagating the exception. ``None`` (default) re-raises so callers see the failure.

    Returns
    -------
    A new dataframe of the same type as ``df`` with the prediction column added. Original ``df`` is NOT mutated.
    """
    if column_name is None:
        t_name = getattr(wrapper, "transform_name", None)
        b_col = getattr(wrapper, "base_column", None)
        if t_name and b_col:
            column_name = f"composite_pred__{t_name}__{b_col}"
        else:
            column_name = "composite_pred"
    try:
        preds = np.asarray(wrapper.predict(df), dtype=np.float64).reshape(-1)
    except Exception:
        if fallback_value is None:
            raise
        n = len(df)
        preds = np.full(n, float(fallback_value), dtype=np.float64)
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        import polars as pl  # lazy
        return df.with_columns(pl.Series(name=column_name, values=preds))
    if isinstance(df, pd.DataFrame):
        out = df.copy()
        out[column_name] = preds
        return out
    raise TypeError(
        f"composite_predictions_as_feature: unsupported df type {type(df).__name__}; pass pandas / polars DataFrame."
    )


def composite_oof_predictions(
    wrapper_factory: Callable[[], Any],
    X: Any,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Out-of-fold composite predictions via K-fold CV.

    Each row's prediction is produced by a wrapper trained on the OTHER folds, so the resulting 1-D array can safely become a feature for a downstream model trained on the same rows -- no in-sample-optimism leakage.

    Parameters
    ----------
    wrapper_factory
        Zero-arg callable that returns a fresh unfitted wrapper instance (e.g. ``lambda: CompositeTargetEstimator(base_estimator=clone(my_lgb), transform_name='linear_residual', base_column='b1')``).
    X
        Feature dataframe (pandas / polars). Rows are split by index.
    y
        Targets (1-D ndarray of length len(X)).
    n_splits
        Number of CV folds (default 5).
    random_state
        Shuffle seed for the KFold splitter.
    fit_kwargs
        Optional dict passed as keyword args to the wrapper's ``fit`` call.

    Returns
    -------
    ``(n,)`` ndarray of OOF predictions on the y-scale. NaN entries indicate a fold that failed to train (caller decides whether to drop / impute).
    """
    from sklearn.model_selection import KFold  # lazy
    fit_kwargs = fit_kwargs or {}
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y_arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    indices = np.arange(n)
    for train_idx, val_idx in kf.split(indices):
        # Subset X for the fold. Polars / pandas handled separately to avoid silent materialisation.
        if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
            import polars as pl  # lazy
            X_train = X.filter(pl.Series([i in set(train_idx.tolist()) for i in range(n)]))
            X_val = X.filter(pl.Series([i in set(val_idx.tolist()) for i in range(n)]))
        elif isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
        else:
            raise TypeError(
                f"composite_oof_predictions: unsupported X type {type(X).__name__}"
            )
        try:
            w = wrapper_factory()
            w.fit(X_train, y_arr[train_idx], **fit_kwargs)
            fold_preds = np.asarray(w.predict(X_val), dtype=np.float64).reshape(-1)
            out[val_idx] = fold_preds
        except Exception as fold_err:
            logger.warning(
                "[composite_oof_predictions] fold failed (val rows %d-%d): %s. NaN-filled.",
                int(val_idx.min()), int(val_idx.max()), fold_err,
            )
    return out
