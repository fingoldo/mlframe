"""Dtype / weight / dataframe-info helpers carved out of ``mlframe.training.extractors``.

Re-imported at the parent module's bottom so historical
``from mlframe.training.extractors import get_sample_weights_by_recency`` import
sites keep working.
"""
from __future__ import annotations

import io
from typing import Dict, Union

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.polarslib import polars_df_info


def get_dataframe_info(df: Union[pd.DataFrame, pl.DataFrame]) -> str:
    """Get a summary info string for a DataFrame.

    Args:
        df: Pandas or Polars DataFrame.

    Returns:
        Info string similar to pandas df.info() output.

    Raises:
        TypeError: If df is not a pandas or Polars DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        buf = io.StringIO()
        df.info(buf=buf, verbose=False)
        return buf.getvalue()
    elif isinstance(df, pl.DataFrame):
        return polars_df_info(df)
    raise TypeError(f"Unsupported DataFrame type: {type(df).__name__}. Expected pandas or Polars DataFrame.")


def _smallest_safe_int_dtype(min_val: int, max_val: int) -> np.dtype:
    """Pick the smallest signed-int numpy dtype that can hold [min_val, max_val].

    Promotion ladder: int8 -> int16 -> int32 -> int64. Multiclass classification
    targets routinely have ``n_classes`` up to thousands (label-encoded
    categorical features, ID-based class assignments); the historical default of
    forcing every classification target to ``int8`` silently wrapped any label
    >127 on the pandas path (``np.array([200]).astype(np.int8) -> -56``) and
    raised hard on the polars path (asymmetric, the trap that surfaced this).
    Polars Series.cast(pl.Int8) raises ``InvalidOperationError``; pandas just
    wraps without warning -- pandas users got scrambled labels downstream.
    """
    if -128 <= min_val and max_val <= 127:
        return np.dtype(np.int8)
    if -32_768 <= min_val and max_val <= 32_767:
        return np.dtype(np.int16)
    if -2_147_483_648 <= min_val and max_val <= 2_147_483_647:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def _safe_int_cast_numpy(arr: np.ndarray, target_name: str) -> np.ndarray:
    """Cast a numpy array to the smallest signed-int dtype that preserves every value."""
    if arr.size == 0:
        return arr.astype(np.int8, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        _min, _max = int(arr.min()), int(arr.max())
    elif np.issubdtype(arr.dtype, np.floating):
        # Refuse fractional floats (e.g. accidentally-numeric target); only allow integer-valued floats.
        if not np.isfinite(arr).all() or not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(
                f"target {target_name!r}: numeric target contains non-integer or non-finite values; "
                f"cannot safely cast to int. Drop NaN/inf rows or use a regression target type."
            )
        _min, _max = int(arr.min()), int(arr.max())
    elif np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int8, copy=False)
    else:
        # Object / string -- pandas/numpy will surface an error from astype itself; let it propagate
        # rather than introducing a separate error path here.
        return arr.astype(np.int8, copy=False)
    _dtype = _smallest_safe_int_dtype(_min, _max)
    if arr.dtype == _dtype:
        return arr
    return arr.astype(_dtype, copy=False)


def intize_targets(targets: Dict[str, Union[pd.Series, pl.Series, np.ndarray]]) -> None:
    """Convert target values to the smallest signed-int numpy dtype that preserves every value.

    Multiclass labels with cardinality >127 used to silently wrap on the pandas
    path (``astype(np.int8)`` is modulo arithmetic for over-range ints) while
    failing loudly on the polars path. Now promotes int8 -> int16 -> int32 -> int64
    based on the actual value range. Multiclass datasets with thousands of
    classes (label-encoded categorical IDs, hash-encoded targets) round-trip
    correctly under this fix.

    Args:
        targets: Dictionary mapping target names to target arrays/series.

    Raises:
        TypeError: If target is not a supported type (pd.Series, pl.Series, np.ndarray).
        ValueError: If a numeric target contains fractional / non-finite values
            (silent truncation hazard).
    """
    for target_name, target in targets.copy().items():
        if isinstance(target, np.ndarray):
            targets[target_name] = _safe_int_cast_numpy(target, target_name)
        elif isinstance(target, pl.Series):
            # to_numpy first (zero-copy on contiguous numeric), then route through the
            # numpy-side range-aware cast so polars + pandas paths share one promotion table.
            targets[target_name] = _safe_int_cast_numpy(target.to_numpy(), target_name)
        elif isinstance(target, pd.Series):
            targets[target_name] = _safe_int_cast_numpy(target.to_numpy(), target_name)
        else:
            raise TypeError(f"Unsupported target type for '{target_name}': {type(target).__name__}")


def get_sample_weights_by_recency(
    date_series: pd.Series,
    min_weight: float = 1.0,
    weight_drop_per_year: float = 0.1,
) -> np.ndarray:
    """Compute sample weights based on recency.

    More recent samples get higher weights. The formula is log-linear
    in days-from-most-recent so that very old samples don't vanish
    entirely while the newest samples get the highest weight.

    Bug fix 2026-04-19: the previous implementation applied
    ``np.log((max - date).days)`` directly. For the single most-recent
    sample (where ``max - date == 0 days``), ``np.log(0) = -inf``, so
    the weight evaluated to ``+inf``. Training-time weighted loss was
    then dominated by that single row (CatBoost/sklearn treat +inf
    weights by clamping or NaN-ing the sample, producing silent fit
    bias that was invisible in the training curve). Also: if all dates
    are identical (``span == 0``), ``np.log(0) -> -inf`` produces an
    all-NaN weight array.

    Now: days-from-max is clipped to ``>= 1`` before the log so the
    newest sample gets a *large finite* weight (floor at
    ``min_weight + max_drop``), and a zero-span series returns uniform
    ``min_weight`` for every row (log-span itself is clipped too).

    Args:
        date_series: Series of datetime values.
        min_weight: Minimum weight for oldest samples.
        weight_drop_per_year: How much weight drops per year of age.

    Returns:
        Array of sample weights (all finite, no NaN / +inf).
    """
    # Use total_seconds() / 86400 instead of .days: ``.days`` floors to integer
    # days and returns 0 for intraday-only datasets (e.g. a single trading day
    # of tick data), which then triggers the uniform-weight branch even though
    # the data has meaningful sub-day age structure.
    #
    # Polymorphic input: callers historically pass a pandas datetime Series
    # (where max - min returns a Timedelta with .total_seconds()) but the
    # FTE also accepts a numeric ts column (int64 / float64 epoch-seconds
    # or any monotone numeric proxy). Numeric (max - min) returns a scalar
    # that has no .total_seconds() method and raises
    # ``AttributeError: 'int' object has no attribute 'total_seconds'``.
    # Detect the numeric path via dtype kind ('i', 'u', 'f') and interpret
    # the raw difference as already-seconds; preserve the datetime path
    # via the original .total_seconds() call.
    _dtype_kind = getattr(getattr(date_series, "dtype", None), "kind", None)
    _is_numeric_ts = _dtype_kind in ("i", "u", "f")
    if _is_numeric_ts:
        # Numeric ts: treat values as seconds-since-some-epoch. max-min is
        # already in seconds; just divide by 86400 to get days.
        span_seconds = float(date_series.max() - date_series.min())
        span_days = span_seconds / 86400.0
    else:
        span_days = (date_series.max() - date_series.min()).total_seconds() / 86400.0
    # Zero-span guard: all dates equal -> uniform weighting. No log needed.
    if span_days <= 0:
        return np.full(len(date_series), float(min_weight))

    # Sub-day resolution preserved via total_seconds(). Floor at one
    # second (~ 1/86400 day) so log never hits zero -- the previous
    # one-day floor erased intraday gradient by clamping every row to
    # log(1)=0.
    if _is_numeric_ts:
        # Numeric path: max-row is a scalar; subtraction is element-wise
        # numeric and the result is already seconds-since-row.
        _delta_secs = np.asarray(date_series.max() - date_series, dtype=np.float64)
    else:
        _delta_secs = (date_series.max() - date_series).dt.total_seconds().to_numpy()
    _min_age_days = 1.0 / 86400.0  # one-second floor
    days_from_max = np.maximum(_delta_secs / 86400.0, _min_age_days)
    # log(span_days) for span<1 day is negative -> max_drop negative.
    # Use log(span_in_seconds) baseline so the gradient stays positive
    # for sub-day spans too.
    max_drop = (np.log(span_days) - np.log(_min_age_days)) * weight_drop_per_year

    sample_weight = (
        min_weight
        + max_drop
        - (np.log(days_from_max) - np.log(_min_age_days)) * weight_drop_per_year
    )

    return sample_weight
