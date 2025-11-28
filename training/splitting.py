"""
Train/validation/test splitting utilities for mlframe.

Provides flexible data splitting with support for:
- Sequential and shuffled splits
- Date-based (whole-day) splitting
- Training set aging limits
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    val_sequential_fraction: Optional[float] = None,
    test_sequential_fraction: Optional[float] = None,
    trainset_aging_limit: Optional[float] = None,
    timestamps: Optional[pd.Series] = None,
    wholeday_splitting: bool = True,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """
    Split data into train, validation, and test sets with flexible sequential/shuffled control.

    Supports three modes:
    1. Date-based splitting (wholeday_splitting=True, timestamps provided)
    2. Row-based splitting with timestamps (wholeday_splitting=False, timestamps provided)
    3. Simple row-based splitting (no timestamps, uses sklearn)

    Args:
        df: Input DataFrame to split.
        test_size: Fraction of data for test set (0.0-1.0).
        val_size: Fraction of remaining data for validation set (0.0-1.0).
        shuffle_val: If True and val_sequential_fraction is None, fully shuffle validation set.
        shuffle_test: If True and test_sequential_fraction is None, fully shuffle test set.
        val_sequential_fraction: Fraction of validation set that should be sequential (0.0-1.0).
            If None and shuffle_val=True: fully shuffled.
            If None and shuffle_val=False: fully sequential.
        test_sequential_fraction: Fraction of test set that should be sequential (0.0-1.0).
            If None and shuffle_test=True: fully shuffled.
            If None and shuffle_test=False: fully sequential.
        trainset_aging_limit: If set, keep only the most recent fraction of training data.
            Must be between 0 and 1.
        timestamps: Series of timestamps for time-based splitting.
        wholeday_splitting: If True and timestamps provided, split by whole days.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_idx, val_idx, test_idx, train_details, val_details, test_details)
        where *_idx are sorted numpy arrays of indices and *_details are description strings.
    """
    if random_seed:
        np.random.seed(random_seed)

    # Validate trainset_aging_limit once at the start (used in multiple branches below)
    if trainset_aging_limit and not (0 < trainset_aging_limit < 1.0):
        raise ValueError(f"trainset_aging_limit must be in (0, 1), got {trainset_aging_limit}")

    def _calculate_split_sizes(total_size, target_size, shuffle, sequential_fraction):
        """Calculate sequential and shuffled portions for a split."""
        n_total = int(total_size * target_size)

        if sequential_fraction is not None:
            if not (0.0 <= sequential_fraction <= 1.0):
                raise ValueError(f"sequential_fraction must be between 0.0 and 1.0, got {sequential_fraction}")
            n_sequential = int(n_total * sequential_fraction)
            n_shuffled = n_total - n_sequential
        else:
            # Legacy behavior: fully shuffled or fully sequential
            n_shuffled = n_total if shuffle else 0
            n_sequential = 0 if shuffle else n_total

        return n_sequential, n_shuffled

    def _perform_split(sorted_items, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf):
        """Perform the actual splitting on sorted items (dates or indices)."""
        remaining = sorted_items.copy()
        test_seq = val_seq = None
        test_list = []
        val_list = []

        # Sequential test (most recent)
        if n_test_seq > 0:
            test_seq = remaining[-n_test_seq:]
            test_list.append(test_seq)
            remaining = remaining[:-n_test_seq]

        # Sequential val (next most recent)
        if n_val_seq > 0:
            val_seq = remaining[-n_val_seq:]
            val_list.append(val_seq)
            remaining = remaining[:-n_val_seq]

        # Shuffled test from remaining
        if n_test_shuf > 0:
            test_shuf_idx = np.random.choice(len(remaining), n_test_shuf, replace=False)
            test_list.append(remaining[test_shuf_idx])
            remaining = np.delete(remaining, test_shuf_idx)

        # Shuffled val from remaining
        if n_val_shuf > 0:
            val_shuf_idx = np.random.choice(len(remaining), n_val_shuf, replace=False)
            val_list.append(remaining[val_shuf_idx])
            remaining = np.delete(remaining, val_shuf_idx)

        test_items = np.concatenate(test_list) if test_list else np.array([], dtype=remaining.dtype)
        val_items = np.concatenate(val_list) if val_list else np.array([], dtype=remaining.dtype)
        train_items = remaining

        return train_items, val_items, test_items, val_seq, test_seq

    def _build_details(timestamps, idx, sequential_idx, n_shuffled, unit) -> str:
        """Build detail string for a split set."""
        if sequential_idx is not None and len(sequential_idx) > 0:
            details = f"{timestamps.iloc[sequential_idx].min():%Y-%m-%d}/{timestamps.iloc[sequential_idx].max():%Y-%m-%d}"
            if n_shuffled > 0:
                details += f" +{n_shuffled} {unit}"
        else:
            if len(idx) > 0:
                details = f"{timestamps.iloc[idx].min():%Y-%m-%d}/{timestamps.iloc[idx].max():%Y-%m-%d}"
            else:
                details = ""
        return details

    # Calculate split sizes
    if wholeday_splitting and timestamps is not None:
        dates = pd.to_datetime(timestamps).dt.date
        unique_dates = dates.unique()
        n_total = len(unique_dates)
    else:
        n_total = len(df)

    n_test_seq, n_test_shuf = _calculate_split_sizes(n_total, test_size, shuffle_test, test_sequential_fraction)
    n_val_seq, n_val_shuf = _calculate_split_sizes(n_total - (n_test_seq + n_test_shuf), val_size, shuffle_val, val_sequential_fraction)

    # Perform splitting
    if wholeday_splitting and timestamps is not None:
        # Date-based splitting
        sorted_dates = np.sort(unique_dates)
        train_dates, val_dates, test_dates, val_dates_seq, test_dates_seq = _perform_split(
            sorted_dates, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf
        )

        # Map dates to row indices
        train_idx = np.where(dates.isin(train_dates))[0]
        val_idx = np.where(dates.isin(val_dates))[0]
        test_idx = np.where(dates.isin(test_dates))[0]

        # Apply aging limit
        if trainset_aging_limit:
            n_dates_to_keep = int(len(train_dates) * trainset_aging_limit)
            if n_dates_to_keep > 0:
                recent_dates = np.sort(train_dates)[-n_dates_to_keep:]
                train_idx = np.where(dates.isin(recent_dates))[0]

        # Build detail strings
        train_details = f"{timestamps.iloc[train_idx].min():%Y-%m-%d}/{timestamps.iloc[train_idx].max():%Y-%m-%d}"

        val_seq_idx = np.where(dates.isin(val_dates_seq))[0] if val_dates_seq is not None else None
        val_details = _build_details(timestamps, val_idx, val_seq_idx, n_val_shuf, "days")

        test_seq_idx = np.where(dates.isin(test_dates_seq))[0] if test_dates_seq is not None else None
        test_details = _build_details(timestamps, test_idx, test_seq_idx, n_test_shuf, "days")

    elif timestamps is not None:
        # Row-based splitting with timestamps
        sorted_idx = np.argsort(timestamps.values)
        train_idx, val_idx, test_idx, val_idx_seq, test_idx_seq = _perform_split(
            sorted_idx, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf
        )

        # Apply aging limit
        if trainset_aging_limit:
            train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)):]

        # Build detail strings
        train_details = f"{timestamps.iloc[train_idx].min():%Y-%m-%d}/{timestamps.iloc[train_idx].max():%Y-%m-%d}"
        val_details = _build_details(timestamps, val_idx, val_idx_seq, n_val_shuf, "records")
        test_details = _build_details(timestamps, test_idx, test_idx_seq, n_test_shuf, "records")

    else:
        # Row-based splitting without timestamps (fallback to sklearn)
        if test_size > 0:
            train_idx, test_idx = train_test_split(
                np.arange(len(df)), test_size=test_size, shuffle=shuffle_test
            )
        else:
            train_idx, test_idx = np.arange(len(df)), None
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size, shuffle=shuffle_val)

        if trainset_aging_limit:
            train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)):]

        train_details, val_details, test_details = "", "", ""

    logger.info(
        f"{len(train_idx):_} train rows {train_details}, "
        f"{len(val_idx):_} val rows {val_details}, "
        f"{len(test_idx) if test_idx is not None else 0:_} test rows {test_details}."
    )

    return (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
        train_details,
        val_details,
        test_details,
    )


__all__ = ["make_train_test_split"]
