"""Global outlier detection helper for ``_setup_helpers``.

Carved from ``_setup_helpers.py`` to keep the parent below the LOC budget.
Re-exported from the parent so historical
``from mlframe.training.core._setup_helpers import _apply_outlier_detection_global``
imports continue to resolve.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from .._ram_helpers import maybe_clean_ram_and_gpu
from ..utils import log_ram_usage

if TYPE_CHECKING:
    from ..configs import (  # noqa: F401
        PreprocessingBackendConfig,
        PreprocessingConfig,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
    )
    from ._training_context import TrainingContext  # noqa: F401

logger = logging.getLogger(__name__)


def _apply_outlier_detection_global(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    outlier_detector: Any,
    od_val_set: bool,
    verbose: bool,
    baseline_rss_mb: float = 0.0,
    df_size_mb: float = 0.0,
    targets_for_classbalance: dict[str, Any] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Apply outlier detection ONCE globally (unsupervised - no target needed)."""
    if outlier_detector is None:
        return train_df, val_df, train_idx, val_idx, None, None

    if verbose:
        logger.info("Fitting outlier detector (once for all targets)...")

    # sklearn outlier detectors coerce input via check_array; non-numeric columns (string/categorical/embedding-list) crash fit. Drop non-numeric on each
    # call so polars and pandas paths stay symmetric. bench-attempt-rejected (2026-05-24): caching the numeric column list across the train + val calls
    # would require asserting train+val schemas match, but val can legitimately have different dtypes (e.g. early-rare-category never seen in train);
    # the per-call schema iteration is ~us on typical column counts so the cache adds maintenance burden without measurable wall gain.
    def _numeric_only_view(df_):
        if isinstance(df_, pl.DataFrame):
            numeric_cols = [
                name for name, dt in df_.schema.items()
                if dt.is_numeric() or dt == pl.Boolean
            ]
            return df_.select(numeric_cols) if len(numeric_cols) != len(df_.columns) else df_
        if hasattr(df_, "select_dtypes"):
            return df_.select_dtypes(include=["number", "bool"])
        return df_

    _train_numeric = _numeric_only_view(train_df)
    # LocalOutlierFactor and OneClassSVM reject NaN inputs (unlike IsolationForest which
    # tolerates NaN on recent sklearn). The mlframe outlier-detection step runs BEFORE
    # the preprocessing pipeline's imputer/fix_infinities, so a NaN-bearing train frame
    # crashes the fit at this line for naive (un-wrapped) detectors. Wrap fit+predict in
    # try/except so the suite degrades gracefully (skips OD + logs the actionable reason)
    # instead of taking down the whole training run. Caller best practice: wrap LOF/OCSVM
    # in ``sklearn.pipeline.Pipeline([SimpleImputer(), detector])`` to keep OD active.
    # Surfaced by fuzz iter#190 (regression x lgb x outlier=lof x NaN-bearing synthetic
    # frame) where a bare ``LocalOutlierFactor`` raised
    # ``ValueError: Input X contains NaN. LocalOutlierFactor does not accept missing values``.
    try:
        outlier_detector.fit(_train_numeric)
        is_inlier = outlier_detector.predict(_train_numeric)
    except (ValueError, TypeError, ImportError, RuntimeError, MemoryError, AttributeError) as _od_exc:
        # Narrowed from bare ``Exception`` so typo/programmer-error attributes raise loudly. The
        # graceful-skip rationale only applies to runtime data issues (NaN inputs, dtype, missing
        # dep, OOM) - not to misconfigured detector classes that should fail fast at fit time.
        logger.error(
            "Outlier detector %s raised during fit/predict on train: %s. Skipping outlier "
            "detection for this run; train_df / val_df returned unfiltered. Wrap the detector "
            "in sklearn.pipeline.Pipeline([SimpleImputer(), %s]) to keep OD active when the "
            "input frame may contain NaN.",
            type(outlier_detector).__name__, _od_exc, type(outlier_detector).__name__,
        )
        return train_df, val_df, train_idx, val_idx, None, None
    train_od_idx = is_inlier == 1

    filtered_train_df = train_df
    filtered_train_idx = train_idx

    def _filter_df_by_mask(_df, mask):
        if isinstance(_df, pl.DataFrame):
            return _df.filter(pl.Series(mask))
        return _df.loc[mask]

    train_kept = train_od_idx.sum()
    if train_kept < len(train_df):
        # When OD is fit on features that include a label-correlated leak feature, the unsupervised
        # detector can flag the rare class as outliers and remove it entirely, leaving train with
        # one unique target and crashing CB/XGB deep in C++. Skip the OD filter when this would
        # happen; fit stays intact for diagnostic logging via train_od_idx.
        _od_destroys_classes = False
        if targets_for_classbalance:
            for _tn, _tv in targets_for_classbalance.items():
                if _tv is None:
                    continue
                try:
                    _y_pre = (
                        _tv[train_idx]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[train_idx]
                    )
                    _y_post = (
                        _tv[train_idx[train_od_idx]]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[train_idx[train_od_idx]]
                    )
                    _arr_pre = np.asarray(_y_pre)
                    _arr_post = np.asarray(_y_post)
                    _flat_pre = _arr_pre.flatten() if _arr_pre.ndim > 1 else _arr_pre
                    _flat_post = _arr_post.flatten() if _arr_post.ndim > 1 else _arr_post
                    if len(np.unique(_flat_pre)) >= 2 and len(np.unique(_flat_post)) < 2:
                        _od_destroys_classes = True
                        logger.error(
                            "Outlier detection would eliminate the entire minority "
                            "class from train target '%s' (pre-OD unique=%d, post-OD "
                            "unique=%d). Typical cause: a feature highly correlated "
                            "with the target (e.g. label-leak feature) drives the "
                            "unsupervised OD to flag the rare class as outliers. "
                            "Skipping OD filter for train; original train_df retained.",
                            _tn,
                            len(np.unique(_flat_pre)),
                            len(np.unique(_flat_post)),
                        )
                        break
                except Exception as _exc:
                    logger.debug("Class-balance pre-check failed for target %s: %s", _tn, _exc)
        if not _od_destroys_classes:
            logger.info("Outlier rejection: %d train samples -> %d kept.", len(train_df), train_kept)
            filtered_train_df = _filter_df_by_mask(train_df, train_od_idx)
            filtered_train_idx = train_idx[train_od_idx]
        else:
            # All-True mask so the downstream polars-fastpath filter is a no-op.
            train_kept = len(train_df)
            train_od_idx = np.ones(len(train_df), dtype=bool)

    # Fail fast on catastrophic misconfiguration (~all samples flagged); otherwise CatBoost/LightGBM
    # crashes 5+ min later with opaque "X is empty" errors.
    min_keep = max(1, int(len(train_df) * 0.01))
    if train_kept < min_keep:
        raise ValueError(
            f"Outlier detector rejected {len(train_df) - train_kept:_} of {len(train_df):_} "
            f"train samples, leaving only {train_kept:_} rows (< {min_keep:_}, 1% of input). "
            f"The detector is likely misconfigured (e.g. contamination too high, trained on "
            f"unrepresentative data, or a sign convention bug). Training cannot proceed."
        )

    filtered_val_df = val_df
    filtered_val_idx = val_idx
    val_od_idx = None

    if val_df is not None and od_val_set:
        # Same NaN-tolerance caveat as the train-side fit: skip the val OD filter if
        # the detector raises on the val frame (e.g. NaN inputs without an imputer
        # wrapper). Train-side OD already succeeded by this point, so don't fail the
        # whole suite on the val-side raise.
        try:
            is_inlier = outlier_detector.predict(_numeric_only_view(val_df))
        except Exception as _od_exc:
            logger.error(
                "Outlier detector %s raised on val frame: %s. Skipping val-side OD filter; "
                "original val_df retained for evaluation.",
                type(outlier_detector).__name__, _od_exc,
            )
            return filtered_train_df, val_df, filtered_train_idx, val_idx, train_od_idx, None
        val_od_idx = is_inlier == 1
        val_kept = val_od_idx.sum()
        # Mirror of train-side class-balance pre-check: skip OD on val if it would wipe out a class.
        if targets_for_classbalance and val_kept < len(val_df) and val_idx is not None:
            for _tn, _tv in targets_for_classbalance.items():
                if _tv is None:
                    continue
                try:
                    _y_pre = (
                        _tv[val_idx]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[val_idx]
                    )
                    _y_post = (
                        _tv[val_idx[val_od_idx]]
                        if isinstance(_tv, (np.ndarray, pl.Series))
                        else _tv.iloc[val_idx[val_od_idx]]
                    )
                    _arr_pre = np.asarray(_y_pre)
                    _arr_post = np.asarray(_y_post)
                    _flat_pre = _arr_pre.flatten() if _arr_pre.ndim > 1 else _arr_pre
                    _flat_post = _arr_post.flatten() if _arr_post.ndim > 1 else _arr_post
                    if len(np.unique(_flat_pre)) >= 2 and len(np.unique(_flat_post)) < 2:
                        logger.error(
                            "Outlier detection would eliminate the entire minority "
                            "class from VAL target '%s' (pre-OD unique=%d, post-OD "
                            "unique=%d). Skipping OD filter for val; original "
                            "val_df retained for evaluation.",
                            _tn,
                            len(np.unique(_flat_pre)),
                            len(np.unique(_flat_post)),
                        )
                        # All-True mask so downstream polars filter is a no-op.
                        val_kept = len(val_df)
                        val_od_idx = np.ones(len(val_df), dtype=bool)
                        break
                except Exception as _exc:
                    logger.debug("Class-balance pre-check on val failed for target %s: %s", _tn, _exc)
        # Symmetric of train-side min_keep guard: don't propagate a near-empty val_df; downstream
        # eval paths handle empty val poorly. Keep original val_df and surface the error.
        val_min_keep = max(1, int(len(val_df) * 0.01))
        if val_kept < val_min_keep:
            logger.error(
                "Outlier detector rejected %d of %d val samples, leaving "
                "only %d rows (< %d, 1%% floor). Continuing with the "
                "ORIGINAL (unfiltered) val_set so downstream evaluation "
                "has data; investigate contamination / fit-distribution "
                "mismatch between train and val.",
                len(val_df) - val_kept, len(val_df), val_kept, val_min_keep,
            )
            filtered_val_df = val_df
            filtered_val_idx = val_idx
            val_od_idx = None
        elif val_kept < len(val_df):
            logger.info("Outlier rejection: %d val samples -> %d kept.", len(val_df), val_kept)
            filtered_val_df = _filter_df_by_mask(val_df, val_od_idx)
            filtered_val_idx = val_idx[val_od_idx]

    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-outlier-detection")
    if verbose:
        log_ram_usage()

    return (filtered_train_df, filtered_val_df, filtered_train_idx, filtered_val_idx, train_od_idx, val_od_idx)
