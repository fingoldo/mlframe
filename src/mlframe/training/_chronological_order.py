"""Order a split's row-index array chronologically, so the frame taken with it is chronological at zero extra memory.

The suite materialises train/val/test with a TAKE by index arrays, and the splitter hands back train positions sorted by ROW NUMBER. When the caller's frame
is not chronological the train rows are not either, which switches off temporal CV folds, CatBoost ``has_time`` and temporal OOF. Sorting the source frame
would duplicate it; reordering only the index array (an O(n_train) int64 vector) before the take gives the same result for free.

Missing timestamps (NaT / NaN / null) go LAST. Ties keep the incoming (row-number) order: the sort is stable.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["naive_utc_series", "timestamps_sort_keys", "chronological_order_index", "apply_chronological_train_order"]

_INT64_MAX = np.iinfo(np.int64).max


def naive_utc_series(series: pd.Series) -> pd.Series:
    """tz-aware values -> UTC-naive; object dtype -> parsed datetimes (unparseable -> NaT); anything else unchanged."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    if series.dtype == object:
        return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    return series


def timestamps_sort_keys(timestamps: Any) -> Optional[np.ndarray]:
    """1-D numeric sort keys for ``timestamps`` (missing -> largest key), or None when they cannot be ordered numerically.

    Only the timestamps vector is touched (never a frame). datetime64 is viewed as int64 without a copy; tz-aware values are
    normalised to UTC; polars Series / pandas Series / numpy / list are all accepted.
    """
    if timestamps is None:
        return None
    if hasattr(timestamps, "to_numpy") and not isinstance(timestamps, (pd.Series, pd.Index)):  # polars Series
        try:
            timestamps = timestamps.to_numpy()
        except Exception as exc:  # an exotic dtype: no keys, the caller keeps the old order
            logger.debug("chronological order skipped: timestamps of type %s have no numpy view (%r)", type(timestamps).__name__, exc)
            return None
    if isinstance(timestamps, (pd.Series, pd.Index)):
        series = pd.Series(timestamps) if isinstance(timestamps, pd.Index) else timestamps
        values = naive_utc_series(series).to_numpy()
    else:
        values = np.asarray(timestamps)
        if values.dtype == object:
            values = pd.to_datetime(pd.Series(values), utc=True, errors="coerce").dt.tz_localize(None).to_numpy()
    if values.ndim != 1:
        return None
    kind = values.dtype.kind
    if kind in "mM":
        keys = values.view(np.int64)
        nat = np.isnat(values)
        return np.where(nat, _INT64_MAX, keys) if nat.any() else keys
    if kind in "iub":
        return np.asarray(values)
    if kind == "f":
        return np.asarray(values)  # NaN sorts last under numpy's stable sort; the sortedness check treats a NaN as a violation via ``_nondecreasing``
    return None


def _nondecreasing(keys: np.ndarray) -> bool:
    """True when ``keys`` is already non-decreasing (NaN, if any, only in a trailing run)."""
    if keys.size < 2:
        return True
    if keys.dtype.kind == "f":
        nan = np.isnan(keys)
        if nan.any():
            first_nan = int(np.argmax(nan))
            if not nan[first_nan:].all():
                return False
            keys = keys[:first_nan]
            if keys.size < 2:
                return True
    return bool(np.all(keys[1:] >= keys[:-1]))


def chronological_order_index(idx: Any, timestamps: Any) -> Tuple[Any, str]:
    """``(ordered_idx, status)``: ``idx`` reordered so ``timestamps[idx]`` is non-decreasing (missing last, ties stable).

    status is one of ``"sorted"`` (already chronological: ``idx`` is returned as the SAME object, no argsort), ``"reordered"``,
    ``"skipped"`` (no timestamps / nothing to order / timestamps not row-aligned or not orderable: ``idx`` returned unchanged).
    """
    if idx is None or timestamps is None:
        return idx, "skipped"
    idx_arr = np.asarray(idx)
    if idx_arr.ndim != 1 or idx_arr.size < 2 or idx_arr.dtype.kind not in "iu":
        return idx, "skipped"
    try:
        n_ts = len(timestamps)
    except TypeError:
        return idx, "skipped"
    if n_ts == 0 or int(idx_arr.max()) >= n_ts or int(idx_arr.min()) < 0:
        return idx, "skipped"
    keys = timestamps_sort_keys(timestamps)
    if keys is None or keys.shape[0] != n_ts:
        return idx, "skipped"
    gathered = keys[idx_arr]
    if _nondecreasing(gathered):
        return idx, "sorted"
    order = np.argsort(gathered, kind="stable")
    return idx_arr[order], "reordered"


def apply_chronological_train_order(train_idx: Any, timestamps: Any, split_config: Any, metadata: dict, verbose: bool = False) -> Any:
    """Split-phase hook: order ``train_idx`` by time when ``split_config.chronological_train_order`` (default on) and timestamps exist.

    Records ``metadata["train_chronological_order"]`` (``sorted`` / ``reordered`` / ``skipped``); with the knob off nothing is touched or recorded.
    """
    if timestamps is None or not bool(getattr(split_config, "chronological_train_order", True)):
        return train_idx
    train_idx, status = chronological_order_index(train_idx, timestamps)
    metadata["train_chronological_order"] = status
    if verbose and status == "reordered":
        logger.info("Train rows ordered chronologically by timestamp (index-only reorder, stable, missing timestamps last; no frame copy).")
    return train_idx
