"""Group-aware temporal cross-validation splitter.

sklearn ships ``TimeSeriesSplit`` (respects time order, ignores groups) and ``GroupKFold`` (isolates groups, ignores
time order). Neither covers time-ordered data that ALSO carries an entity/group key (customer, session, device):
GroupKFold would happily train on a future group and test on a past one, inflating the CV score on any
non-stationary signal. ``GroupTimeSeriesSplit`` fills that gap - it forward-chains at the GROUP level:

  * groups are ordered by first appearance (the temporal proxy the RFECV temporal auto-detect already relies on: a
    monotonic time axis means row order is time order, so first-appearance order is group time order);
  * each fold trains on an earlier contiguous block of groups and tests on the next block, so every test group is
    strictly later than every train group (temporal guarantee);
  * a group never straddles the train/test boundary (entity-isolation guarantee).

This is the group-level analogue of ``TimeSeriesSplit``; passing ``groups`` where each row is its own group
reproduces ``TimeSeriesSplit`` exactly.
"""
from __future__ import annotations

import numpy as np


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
        return self.n_splits

    @staticmethod
    def _ordered_unique(groups: np.ndarray) -> np.ndarray:
        """Unique group labels in order of FIRST appearance (row = time proxy). pd.unique preserves that order;
        np.unique would sort lexically and destroy the temporal ordering."""
        try:
            import pandas as pd

            return pd.unique(np.asarray(groups))
        except Exception:
            g = np.asarray(groups)
            _, idx = np.unique(g, return_index=True)
            return g[np.sort(idx)]

    def split(self, X=None, y=None, groups=None):
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
        return (
            f"GroupTimeSeriesSplit(n_splits={self.n_splits}, "
            f"max_train_groups={self.max_train_groups}, gap={self.gap})"
        )
