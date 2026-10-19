"""Stages of ``_make_train_test_sp_wholeday_splitting_timestamps_none``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.training._splitting_helpers import _build_details
from mlframe.training._splitting_helpers import _perform_split
from mlframe.training._splitting_helpers import fmt_ts
from mlframe.training.splitting import _make_train_test_sp_consistent_row_timestamp_branch
from mlframe.training.splitting import _make_train_test_sp_lbl_split_dates_train
from mlframe.training.splitting import logger
# --- end imports ---


def _split_by_wholeday(unique_dates, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, _effective_val_placement, trainset_aging_limit, dates, timestamps, test_idx, train_idx, val_idx):
    """Stage of ``_make_train_test_sp_wholeday_splitting_timestamps_none``."""
    # Date-based splitting
    sorted_dates = np.sort(unique_dates)
    train_dates, val_dates, test_dates, val_dates_seq, test_dates_seq, eff_test_shuf, eff_val_shuf = _perform_split(
        sorted_dates, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, _effective_val_placement
    )

    # Apply aging limit BEFORE computing train_idx / train_details,
    # so the printed train date range reflects the actually-used rows
    # (consistent with the row-timestamp branch below).
    train_dates = _make_train_test_sp_consistent_row_timestamp_branch(trainset_aging_limit, train_dates)

    # Map dates -> split label once, then derive all index arrays from the cached
    # label array. ``dates.map(dict)`` triggers a per-row Python apply path
    # on pandas Series of datetime values when the lookup dict is small,
    # measured ~6x slower than factorize+gather on a 1M-row daily span.
    # We factorize once to get integer codes per unique date, then build
    # a (n_unique,)-sized label LUT and gather labels[codes] in pure NumPy.
    # Label convention: 0=train, 1=val, 2=test, -1=dropped by aging.
    codes, uniq_index = pd.factorize(dates, sort=False)
    # uniq_index is a pandas Index of unique dates in first-occurrence
    # order; build a O(1) lookup from datetime -> position via a dict.
    # normalise keys + lookups to int64 nanoseconds so the
    # ``dict.get(d)`` calls below work regardless of whether ``train_dates``
    # is a numpy datetime64 array (default on most pandas versions) or a
    # pandas DatetimeIndex of Timestamps (pandas 3.0 path). The previous
    # ``{d: i for i, d in enumerate(uniq_index)}`` keyed by Timestamps
    # silently lost lookups when ``train_dates`` was np.datetime64 because
    # Timestamp.__hash__ != np.datetime64.__hash__ on pandas 3.0; every
    # label_lut entry stayed -1 and every split came out empty.
    def _as_ns_int(_d):
        """Normalises a single date-like value to an int64-nanosecond key, falling back to the raw value if it isn't Timestamp-convertible."""
        try:
            return int(pd.Timestamp(_d).value)
        except (ValueError, TypeError, OverflowError) as _exc:
            logger.debug("_as_ns_int fell back to its default after %s", type(_exc).__name__)
            return _d

    def _ns_keys(_arr):
        """Batch-converts an array of date-likes to int64-ns keys via ``DatetimeIndex.asi8``, falling back to per-element ``_as_ns_int`` for non-datetime-convertible arrays."""
        # Vectorised datetime -> int64-ns keys. ``pd.DatetimeIndex(...).asi8``
        # is the batch equivalent of the per-element ``int(pd.Timestamp(_d).value)``
        # (verified identical), ~10x faster on a 50k-date span by avoiding one
        # Timestamp construction per date across uniq_index + train/val/test. The
        # per-element ``_as_ns_int`` path is kept as the fallback for arrays that
        # are not datetime-convertible: DatetimeIndex raises on any unconvertible
        # element, so a successful conversion implies every element matches
        # pd.Timestamp, and the fallback then reproduces the old value-as-is keys.
        try:
            return pd.DatetimeIndex(np.asarray(_arr)).asi8.tolist()
        except (ValueError, TypeError, OverflowError) as _exc:
            logger.debug("_ns_keys fell back to its default after %s", type(_exc).__name__)
            return [_as_ns_int(_d) for _d in _arr]

    date_to_code = {k: i for i, k in enumerate(_ns_keys(uniq_index))}
    label_lut = np.full(len(uniq_index), -1, dtype=np.int8)
    _make_train_test_sp_lbl_split_dates_train(train_dates, val_dates, test_dates, _ns_keys, date_to_code, label_lut)
    # `codes` already aligned to `dates` order; gather is a single C-loop.
    # Rows whose date was outside any of train/val/test dates retain -1
    # (codes == -1 from factorize for unknown values; clamp via where).
    labels = np.where(codes >= 0, label_lut[codes], -1)
    train_idx = np.flatnonzero(labels == 0)
    val_idx = np.flatnonzero(labels == 1)
    test_idx = np.flatnonzero(labels == 2)

    def _dates_to_idx(dates_subset):
        """Maps a subset of dates to row positional indices via the shared ``date_to_code``/``codes`` LUT built above; returns ``None`` when ``dates_subset`` is ``None`` (no sequential portion requested)."""
        if dates_subset is None:
            return None
        subset_mask = np.zeros(len(uniq_index), dtype=bool)
        for d in dates_subset:
            # Use the same int64-ns normalisation as the label-LUT loops
            # so dict lookups work regardless of whether `dates_subset`
            # is a numpy datetime64 array or a pandas DatetimeIndex.
            _i = date_to_code.get(_as_ns_int(d))
            if _i is not None:
                subset_mask[_i] = True
        return np.flatnonzero(np.where(codes >= 0, subset_mask[codes], False))

    val_seq_idx = _dates_to_idx(val_dates_seq)
    test_seq_idx = _dates_to_idx(test_dates_seq)

    # Build detail strings. ``.min()`` on empty index yields NaT, which
    # then crashes the ``:%Y-%m-%d`` formatter. Guard for empty train.
    if len(train_idx) > 0:
        train_details = f"{fmt_ts(timestamps.iloc[train_idx].min())}/{fmt_ts(timestamps.iloc[train_idx].max())}"
    else:
        train_details = "(empty)"

    val_details = _build_details(timestamps, val_idx, val_seq_idx, eff_val_shuf, "D")
    test_details = _build_details(timestamps, test_idx, test_seq_idx, eff_test_shuf, "D")
    return test_details, test_idx, train_details, train_idx, val_details, val_idx
