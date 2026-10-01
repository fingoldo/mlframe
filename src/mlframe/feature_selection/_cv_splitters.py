"""Row-order-independent CV splitters shared by RFECV and the feature-selector split policy (``feature_selection.cv_policy``).

``GroupTimeSeriesSplit`` forward-chains over time-ordered groups; ``TimestampOrderedSplit`` takes time order from an explicit timestamp array instead of
row position. This module imports only numpy / pandas / sklearn so any selector can use it without pulling in the RFECV package.

sklearn ships ``TimeSeriesSplit`` (respects time order, ignores groups) and ``GroupKFold`` (isolates groups, ignores time order). Neither covers time-ordered
data that ALSO carries an entity key: GroupKFold would train on a future group and test on a past one. ``GroupTimeSeriesSplit`` forward-chains at the GROUP
level (groups ordered by first appearance; every test group strictly later than every train group; a group never straddles the boundary), and with each row
its own group reproduces ``TimeSeriesSplit`` exactly.

``TimeSeriesSplit`` and ``GroupTimeSeriesSplit`` read row position as time, which is only right when the frame is sorted by time. The training suite keeps
the caller's row order (its chronological main split only decides which rows land in train / val / test), so on a frame that is not sorted by its timestamp
column a positional splitter trains on future rows and scores on past ones. ``TimestampOrderedSplit`` sorts by the supplied timestamps and forward-chains
in that order.

Fold indices come back in chronological order rather than ascending position, so ``X.iloc[train_idx]`` is itself time-ordered and RFECV's nested
early-stopping split over it (``early_stopping_val_cv``) holds out the newest rows.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


class GroupTimeSeriesSplit:
    """Forward-chaining CV over time-ordered groups (entity isolation + temporal order).

    Parameters
    ----------
    n_splits : int, default 5
        Number of splits. Requires at least ``n_splits + 1`` distinct groups.
    max_train_groups : int or None, default None
        Cap on the number of most-recent groups in each training block (rolling window). ``None`` = expanding
        window (all earlier groups), matching ``TimeSeriesSplit``'s default.
    gap : int, default 0
        Number of groups to skip between the train block and the test block (embargo), to blunt leakage from
        autocorrelation across the boundary.

    Notes
    -----
    Group time order is taken from FIRST APPEARANCE in ``groups`` (row order). On a monotonic time axis this is the
    true temporal order; if the rows are not time-sorted, sort them (or the frame) by time before calling.
    """

    def __init__(self, n_splits: int = 5, max_train_groups: int | None = None, gap: int = 0) -> None:
        if n_splits < 1:
            raise ValueError(f"GroupTimeSeriesSplit: n_splits must be >= 1; got {n_splits}.")
        if gap < 0:
            raise ValueError(f"GroupTimeSeriesSplit: gap must be >= 0; got {gap}.")
        if max_train_groups is not None and max_train_groups < 1:
            raise ValueError(f"GroupTimeSeriesSplit: max_train_groups must be >= 1 or None; got {max_train_groups}.")
        self.n_splits = int(n_splits)
        self.max_train_groups = None if max_train_groups is None else int(max_train_groups)
        self.gap = int(gap)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds this splitter yields (sklearn CV-splitter protocol); ``X``/``y``/``groups`` are accepted but unused."""
        return self.n_splits

    @staticmethod
    def _ordered_unique(groups: np.ndarray) -> np.ndarray:
        """Unique group labels in order of FIRST appearance (row = time proxy). pd.unique preserves that order;
        np.unique would sort lexically and destroy the temporal ordering."""
        try:
            import pandas as pd

            return np.asarray(pd.unique(np.asarray(groups)))
        except Exception as e:
            logger.debug("pandas unique() group-order extraction failed, falling back to np.unique: %s", e)
            g = np.asarray(groups)
            _, idx = np.unique(g, return_index=True)
            return g[np.sort(idx)]

    def split(self, X=None, y=None, groups=None):
        """Yield ``(train_idx, test_idx)`` row-index arrays for each fold, forward-chaining over ``groups`` in first-appearance order.

        ``X``/``y`` are accepted (sklearn CV-splitter protocol) but unused; ``groups`` is required. Folds whose train block is
        fully consumed by ``gap`` (too early in the sequence) or whose train/test block ends up empty are silently skipped."""
        if groups is None:
            raise ValueError("GroupTimeSeriesSplit requires groups.")
        groups = np.asarray(groups)
        n_samples = groups.shape[0]
        ordered = self._ordered_unique(groups)
        n_groups = ordered.shape[0]
        n_folds = self.n_splits + 1
        if n_groups < n_folds:
            raise ValueError(
                f"GroupTimeSeriesSplit: {n_groups} distinct groups is too few for n_splits={self.n_splits} "
                f"(need at least {n_folds}). Reduce n_splits or provide more groups."
            )
        # Position of each group in the temporal order (0 = earliest); vectorised row -> group-order lookup.
        order_of = {g: i for i, g in enumerate(ordered)}
        group_pos = np.fromiter((order_of[g] for g in groups), dtype=np.int64, count=n_samples)

        # Same block arithmetic as sklearn TimeSeriesSplit, applied to the GROUP axis.
        test_size = n_groups // n_folds
        test_starts = range(n_groups - self.n_splits * test_size, n_groups, test_size)
        all_rows = np.arange(n_samples)
        for test_start in test_starts:
            train_end = test_start - self.gap
            if train_end <= 0:
                # gap consumed the entire train block for this early fold; skip it rather than yield an empty train.
                continue
            train_lo = 0 if self.max_train_groups is None else max(0, train_end - self.max_train_groups)
            train_mask = (group_pos >= train_lo) & (group_pos < train_end)
            test_mask = (group_pos >= test_start) & (group_pos < test_start + test_size)
            train_idx = all_rows[train_mask]
            test_idx = all_rows[test_mask]
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx

    def __repr__(self) -> str:
        return f"GroupTimeSeriesSplit(n_splits={self.n_splits}, max_train_groups={self.max_train_groups}, gap={self.gap})"


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

    ``n_splits`` is the number of forward-chained folds. ``timestamps`` is one timestamp per row of the ``X`` later passed to ``split``; ``None`` means
    the rows are already in time order. The array is held by reference, never copied, and is not pickled (a fitted selector only needs the folds it
    already produced); a split on a length mismatch, including after unpickling, falls back to row order with a warning. ``gap``, ``max_train_size`` and
    ``test_size`` are forwarded to ``TimeSeriesSplit`` on the groups-free path. ``groups`` are row-aligned group labels bound to the splitter for callers
    that never pass ``groups`` to ``split`` (``cross_val_score`` without ``groups=``); an explicit ``groups`` argument to ``split`` wins, and they are
    held by reference and not pickled, like ``timestamps``.

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
        groups: Any = None,
    ) -> None:
        if int(n_splits) < 2:
            raise ValueError(f"TimestampOrderedSplit: n_splits must be >= 2; got {n_splits}.")
        self.n_splits = int(n_splits)
        self.timestamps = timestamps
        self.gap = int(gap)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.groups = groups

    # Folds depend only on the row order, never on a seed, so a caller that would otherwise repeat the CV under different seeds can skip that.
    deterministic_folds = True
    _timestamps_dropped = False

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
        if groups is None and self.groups is not None and len(self.groups) == n:
            groups = self.groups
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
        for tr, te in GroupTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap).split(groups=groups_in_time_order):
            yield order[tr], order[te]

    def __deepcopy__(self, memo: dict) -> "TimestampOrderedSplit":
        """Share the timestamp array: ``sklearn.clone`` deep-copies constructor params, and a per-clone copy of a
        train-length array is pure RAM waste since the splitter never mutates it."""
        return TimestampOrderedSplit(
            n_splits=self.n_splits, timestamps=self.timestamps, gap=self.gap, max_train_size=self.max_train_size, test_size=self.test_size,
            groups=self.groups,
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_timestamps_dropped"] = self.timestamps is not None
        state["timestamps"] = None
        state["groups"] = None
        return state

    def __repr__(self) -> str:
        n_ts = None if self.timestamps is None else len(self.timestamps)
        return (
            f"TimestampOrderedSplit(n_splits={self.n_splits}, timestamps=<{n_ts} values>, gap={self.gap}, "
            f"max_train_size={self.max_train_size}, test_size={self.test_size})"
        )
