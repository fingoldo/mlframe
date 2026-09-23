"""Composite x FE-pipeline stacking helpers: ``composite_predictions_as_feature`` attaches a fitted wrapper's predictions to a dataframe as a new column; ``composite_oof_predictions`` runs K-fold OOF stacking to avoid in-sample leakage. Both use ``CompositeTargetEstimator``-style wrappers passed in by the caller; no compile-time dep on the wrapper class itself."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from mlframe.utils.log_throttle import log_throttle

from .._frame_ops import append_column

logger = logging.getLogger(__name__)

# Kept for callers that still pass ``allow_large_frame_copy``: appending through ``append_column`` shares the existing
# columns on every supported pandas, so there is no large-frame copy left to gate.
_FEATURE_STACK_LARGE_FRAME_BYTES: int = 2 * 1024**3


# ----------------------------------------------------------------------
# composite_predictions_as_feature (composite x FE-pipeline stacking).
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
    column_name: str | None = None,
    fallback_value: float | None = None,
    allow_large_frame_copy: bool = False,
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
    allow_large_frame_copy
        Accepted and ignored. The append shares the source frame's columns instead of copying them, so no frame size needs the caller's permission.

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
    except Exception as exc:
        if fallback_value is None:
            raise
        logger.warning(
            "composite_predictions_as_feature: wrapper.predict failed for column '%s'; filling with the constant "
            "fallback_value=%s instead. This masks any real predict-time bug (schema drift, shape mismatch, stale "
            "wrapper) as an intentional abstain -- verify this is expected. Error: %s",
            column_name, fallback_value, exc,
        )
        n = len(df)
        preds = np.full(n, float(fallback_value), dtype=np.float64)
    if isinstance(df, (pd.DataFrame,)) or type(df).__module__.startswith("polars"):
        return append_column(df, column_name, preds)
    raise TypeError(f"composite_predictions_as_feature: unsupported df type {type(df).__name__}; pass pandas / polars DataFrame.")


def composite_oof_predictions(
    wrapper_factory: Callable[[], Any],
    X: Any,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    fit_kwargs: dict[str, Any] | None = None,
    time_aware: bool = False,
    cv_splitter: Any = None,
    groups: np.ndarray | None = None,
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
        Optional dict passed as keyword args to the wrapper's ``fit`` call. A per-row ``sample_weight`` of length ``len(X)`` is sliced PER FOLD to the fold-train rows so the wrapper never sees a length-mismatched weight vector (forwarding it verbatim would mis-align weights to rows or raise). Any other ``fit_kwargs`` entry is passed through unchanged.
    groups
        Optional ``(n,)`` group labels for group-aware OOF. When supplied (and no explicit ``cv_splitter`` is given), the splitter defaults to ``GroupKFold`` and the labels are forwarded to ``split(...)`` so a row's fold-train wrappers never see another row from the same group. ``groups`` is also forwarded to any caller-supplied ``cv_splitter`` -- required by ``GroupKFold`` / ``StratifiedGroupKFold`` / ``GroupShuffleSplit`` (their ``split`` raises when ``groups`` is ``None``), ignored by ``KFold`` / ``TimeSeriesSplit``.

    Returns
    -------
    ``(n,)`` ndarray of OOF predictions on the y-scale. NaN entries indicate a fold that failed to train (caller decides whether to drop / impute).
    """
    from sklearn.model_selection import GroupKFold, TimeSeriesSplit  # lazy
    fit_kwargs = fit_kwargs or {}
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y_arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    groups_arr = None if groups is None else np.asarray(groups).reshape(-1)
    if groups_arr is not None and groups_arr.size != n:
        raise ValueError(f"composite_oof_predictions: groups length {groups_arr.size} != n {n}.")
    # A full-length per-row sample_weight must be sliced to each fold's train rows; mark it so the per-fold loop slices it (and leave any wrapper-already-fold-scoped fit_kwargs untouched).
    _sw_full = None
    _sw = fit_kwargs.get("sample_weight", None)
    if _sw is not None:
        _sw_arr = np.asarray(_sw)
        if _sw_arr.reshape(-1).shape[0] == n:
            _sw_full = _sw_arr.reshape(-1)
    if cv_splitter is not None:
        kf = cv_splitter
    elif groups_arr is not None:
        kf = GroupKFold(n_splits=int(n_splits))
    elif time_aware:
        kf = TimeSeriesSplit(n_splits=int(n_splits))
    else:
        from ..discovery._splitter import make_discovery_splitter  # the one place a shuffled discovery KFold is built

        kf = make_discovery_splitter(int(n_splits), random_state=int(random_state))[0]
    indices = np.arange(n)
    try:
        import polars as pl
        _HAS_POLARS = True
    except ImportError:
        pl = None  # type: ignore[assignment]
        _HAS_POLARS = False
    for train_idx, val_idx in kf.split(indices, y_arr, groups_arr):
        # Subset X for the fold. Polars / pandas handled separately to avoid silent materialisation.
        if _HAS_POLARS and isinstance(X, pl.DataFrame):
            # Gather by position (order-preserving, like the pandas .iloc branch); a boolean mask keeps row order instead.
            X_train = X[np.asarray(train_idx, dtype=np.int64)]
            X_val = X[np.asarray(val_idx, dtype=np.int64)]
        elif isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
        else:
            raise TypeError(f"composite_oof_predictions: unsupported X type {type(X).__name__}")
        # Slice a full-length per-row sample_weight to THIS fold's train rows; a verbatim length-n vector would mis-align to the len(train_idx) rows the wrapper fits on (or raise).
        if _sw_full is None:
            fold_fit_kwargs = fit_kwargs
        else:
            fold_fit_kwargs = dict(fit_kwargs)
            fold_fit_kwargs["sample_weight"] = _sw_full[train_idx]
        try:
            w = wrapper_factory()
            w.fit(X_train, y_arr[train_idx], **fold_fit_kwargs)
            fold_preds = np.asarray(w.predict(X_val), dtype=np.float64).reshape(-1)
            out[val_idx] = fold_preds
        except Exception as fold_err:
            log_throttle(
                logger,
                "composite_oof_predictions_fold_failed",
                logging.WARNING,
                "[composite_oof_predictions] fold failed (val rows %d-%d): %s. NaN-filled.",
                int(val_idx.min()), int(val_idx.max()), fold_err,
            )
    return out
