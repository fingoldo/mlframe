"""Give the suite's RFECV selectors a time-ordered CV when the suite's own split is chronological.

The suite knows the rows are temporal whenever the features-and-targets extractor returns ``timestamps`` (its
``ts_field``): the main split then sorts by them and carves val / test from the newest rows. RFECV never saw that
signal, because the timestamp column is normally in ``columns_to_drop`` and the suite passed no hint, so its inner CV
shuffled freely across time and chose features on future-into-past folds that the chronological test cannot reward.

The train rows keep the caller's row order, so ``TimeSeriesSplit`` over them would only be right on a frame already
sorted by time. The folds come from ``TimestampOrderedSplit`` holding the train rows' own timestamps instead.

Temporal unless the caller said otherwise, checked in this order:
    * ``hyperparams_config.has_time`` set explicitly wins either way (it is the same switch CatBoost's ``has_time`` reads);
    * no timestamps -> not temporal;
    * an explicit ``cv`` or ``cv_shuffle=True`` in ``hyperparams_config.rfecv_kwargs`` -> left to the caller;
    * a split whose val and test are both drawn entirely at random -> not temporal: the caller asked for an i.i.d. estimate;
    * otherwise temporal.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_selection.cv_policy import TimestampOrderedSplit, cv_is_temporal

logger = logging.getLogger(__name__)

_RFECV_DEFAULT_N_SPLITS = 3


def rfecv_cv_is_temporal(timestamps: Any, split_config: Any, hyperparams_config: Any) -> Tuple[bool, str]:
    """``(temporal, reason)`` for the suite's RFECV CV: the shared decision plus RFECV's own explicit-``cv`` / ``cv_shuffle`` opt-outs."""
    fields_set: set = set(getattr(hyperparams_config, "model_fields_set", None) or ())
    user_kwargs = (getattr(hyperparams_config, "rfecv_kwargs", None) or {}) if "rfecv_kwargs" in fields_set else {}
    override: Optional[str] = None
    if user_kwargs.get("cv") is not None:
        override = "hyperparams_config.rfecv_kwargs sets cv explicitly"
    elif user_kwargs.get("cv_shuffle") is True:
        override = "hyperparams_config.rfecv_kwargs sets cv_shuffle=True"
    return cv_is_temporal(timestamps, split_config, hyperparams_config, override)


def _auto_n_splits(cv: Any) -> Optional[int]:
    """Fold count of a ``cv`` the suite may replace (None / int / plain ``TimeSeriesSplit``), else None for a caller's own splitter."""
    if cv is None:
        return _RFECV_DEFAULT_N_SPLITS
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        return int(cv)
    if type(cv) is TimeSeriesSplit:
        return int(cv.n_splits)
    return None


def _train_timestamps(timestamps: Any, train_idx: Optional[np.ndarray]) -> Any:
    """Timestamps of the rows RFECV is fit on, keeping a pandas Series (a tz-aware one turns into objects through np.asarray)."""
    if train_idx is None:
        return timestamps
    if hasattr(timestamps, "iloc"):
        return timestamps.iloc[np.asarray(train_idx)]
    return np.asarray(timestamps)[np.asarray(train_idx)]


def apply_temporal_cv_to_rfecv(
    rfecv_models_params: dict,
    *,
    timestamps: Any,
    train_idx: Optional[np.ndarray],
    split_config: Any,
    hyperparams_config: Any,
    verbose: Any = 0,
) -> bool:
    """Swap each suite-built RFECV's CV for ``TimestampOrderedSplit`` over the train rows when the suite is temporal.

    Only a CV the suite chose itself is replaced (``None``, an int, or a plain ``TimeSeriesSplit``, which on unsorted rows
    chains in row order rather than time order); a caller's own splitter is kept. Returns whether anything was replaced.
    """
    if not rfecv_models_params:
        return False
    temporal, reason = rfecv_cv_is_temporal(timestamps, split_config, hyperparams_config)
    if not temporal or timestamps is None:
        if verbose and timestamps is not None:
            logger.info("RFECV CV stays non-temporal: %s.", reason)
        return False
    train_ts = _train_timestamps(timestamps, train_idx)
    replaced = False
    for name, rfecv in rfecv_models_params.items():
        if rfecv is None or not hasattr(rfecv, "cv"):
            continue
        n_splits = _auto_n_splits(rfecv.cv)
        if n_splits is None:
            continue
        old_cv = rfecv.cv
        tss_kwargs = {"gap": old_cv.gap, "max_train_size": old_cv.max_train_size, "test_size": old_cv.test_size} if isinstance(old_cv, TimeSeriesSplit) else {}
        rfecv.cv = TimestampOrderedSplit(n_splits=n_splits, timestamps=train_ts, **tss_kwargs)
        rfecv.cv_shuffle = False
        replaced = True
        if verbose:
            logger.info("%s: using %s because %s (was cv=%r).", name, rfecv.cv, reason, old_cv)
    return replaced
