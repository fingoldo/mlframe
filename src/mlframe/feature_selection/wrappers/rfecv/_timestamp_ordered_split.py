"""Forward-chaining CV that takes time order from an explicit timestamp array instead of from row position.

``TimeSeriesSplit`` and ``GroupTimeSeriesSplit`` read row position as time, which is only right when the frame is sorted
by time. The training suite keeps the caller's row order (its chronological main split only decides which rows land in
train / val / test), so on a frame that is not sorted by its timestamp column a positional splitter trains on future rows
and scores on past ones. ``TimestampOrderedSplit`` sorts by the supplied timestamps and forward-chains in that order.

Fold indices come back in chronological order rather than ascending position, so ``X.iloc[train_idx]`` is itself
time-ordered and RFECV's nested early-stopping split over it (``early_stopping_val_cv``) holds out the newest rows.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from ._group_time_series_split import GroupTimeSeriesSplit

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _n_rows(X: Any, y: Any, groups: Any) -> int:
    """Row count from the first of ``X`` / ``y`` / ``groups`` that has one."""
    for obj in (X, y, groups):
        if obj is None:
            continue
        shape = getattr(obj, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        return len(obj)
    raise ValueError("TimestampOrderedSplit.split needs at least one of X, y, groups.")


def chronological_order(timestamps: Any) -> np.ndarray:
    """Stable argsort of ``timestamps``; missing values (NaT / NaN) sort last, ties keep their row order."""
    series = timestamps if isinstance(timestamps, pd.Series) else pd.Series(np.asarray(timestamps))
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    elif series.dtype == object:
        series = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    values = series.to_numpy()
    if values.dtype.kind in "mM":
        keys = values.view(np.int64).copy()
        keys[np.isnat(values)] = np.iinfo(np.int64).max
        return np.argsort(keys, kind="stable")
    return np.argsort(values, kind="stable")


class TimestampOrderedSplit:
    """``TimeSeriesSplit`` over rows sorted by ``timestamps``; with ``groups``, ``GroupTimeSeriesSplit`` over the same order.

    Parameters
    ----------
    n_splits : int, default 5
        Number of forward-chained folds.
    timestamps : array-like or None
        One timestamp per row of the ``X`` later passed to ``split``. ``None`` means the rows are already in time order.
        The array is held by reference, never copied, and is not pickled (a fitted selector only needs the folds it
        already produced); a split on a length mismatch, including after unpickling, falls back to row order with a
        warning.
    gap, max_train_size, test_size
        Forwarded to ``TimeSeriesSplit`` on the groups-free path.

    With ``groups``, a group's time is its earliest timestamp, and when there are fewer than ``n_splits + 1`` distinct
    groups the split falls back to ``GroupKFold`` (entity isolation without time order), as RFECV's own group path does.
    """

    def __init__(
        self,
        n_splits: int = 5,
        timestamps: Any = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ) -> None:
        if int(n_splits) < 2:
            raise ValueError(f"TimestampOrderedSplit: n_splits must be >= 2; got {n_splits}.")
        self.n_splits = int(n_splits)
        self.timestamps = timestamps
        self.gap = int(gap)
        self.max_train_size = max_train_size
        self.test_size = test_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds this splitter yields (sklearn CV-splitter protocol)."""
        return self.n_splits

    def early_stopping_val_cv(self, n_splits: int) -> "TimestampOrderedSplit":
        """Splitter for the early-stopping hold-out inside one fold's train rows, which ``split`` already returns in time order."""
        return TimestampOrderedSplit(n_splits=n_splits, gap=self.gap)

    def _order(self, n: int) -> np.ndarray:
        """Row positions in chronological order, or ``arange(n)`` when no usable timestamps are held."""
        if self.timestamps is None:
            if getattr(self, "_timestamps_dropped", False):
                logger.warning(
                    "TimestampOrderedSplit: its timestamps were not pickled; using row order as time order, which is only "
                    "correct when the rows are already sorted by time. Rebuild the splitter with timestamps to refit.",
                )
            return np.arange(n)
        n_ts = len(self.timestamps)
        if n_ts != n:
            logger.warning(
                "TimestampOrderedSplit: holds %d timestamps but was asked to split %d rows; using row order as time order, "
                "which is only correct when the rows are already sorted by time.",
                n_ts, n,
            )
            return np.arange(n)
        return chronological_order(self.timestamps)

    def split(self, X: Any = None, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs; every test row is later than every train row of its fold."""
        n = _n_rows(X, y, groups)
        order = self._order(n)
        if groups is None:
            inner = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap, max_train_size=self.max_train_size, test_size=self.test_size)
            for tr, te in inner.split(order):
                yield order[tr], order[te]
            return
        groups_in_time_order = np.asarray(groups)[order]
        n_groups = pd.unique(groups_in_time_order).shape[0]
        if n_groups < self.n_splits + 1:
            logger.warning(
                "TimestampOrderedSplit: only %d distinct groups for n_splits=%d (need >= %d); falling back to GroupKFold, "
                "so folds isolate groups but are not ordered in time.",
                n_groups, self.n_splits, self.n_splits + 1,
            )
            yield from GroupKFold(n_splits=min(self.n_splits, n_groups)).split(order, groups=np.asarray(groups))
            return
        for tr, te in GroupTimeSeriesSplit(n_splits=self.n_splits).split(groups=groups_in_time_order):
            yield order[tr], order[te]

    def __deepcopy__(self, memo: dict) -> "TimestampOrderedSplit":
        """Share the timestamp array: ``sklearn.clone`` deep-copies constructor params, and a per-clone copy of a
        train-length array is pure RAM waste since the splitter never mutates it."""
        return TimestampOrderedSplit(
            n_splits=self.n_splits, timestamps=self.timestamps, gap=self.gap, max_train_size=self.max_train_size, test_size=self.test_size,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_timestamps_dropped"] = self.timestamps is not None
        state["timestamps"] = None
        return state

    def __repr__(self) -> str:
        n_ts = None if self.timestamps is None else len(self.timestamps)
        return (
            f"TimestampOrderedSplit(n_splits={self.n_splits}, timestamps=<{n_ts} values>, gap={self.gap}, "
            f"max_train_size={self.max_train_size}, test_size={self.test_size})"
        )
