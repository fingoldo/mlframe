"""Pure split-arithmetic helpers carved from :func:`make_train_test_split`.

Lifted out of ``splitting.py`` to keep that module under the 1k-LOC ceiling. These were nested
closures in ``make_train_test_split``; the two closure references (``rng`` and the resolved
``effective_val_placement``) are now explicit parameters so the functions are pure module-level.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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

def _perform_split(sorted_items, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, effective_val_placement):
    """Perform the actual splitting on sorted items (dates or indices).

    With ``effective_val_placement == "backward"`` the sequential val
    slice is taken from the OLDEST end of ``remaining`` after the test
    slice is removed, producing the Mazzanti layout
    ``[val(oldest)] [train(middle)] [test(newest)]``. Forward (default)
    takes it from the newest end of the same remainder, producing
    ``[train(oldest)] [val(middle)] [test(newest)]``. The shuffled
    portions are drawn from the remainder after BOTH sequential slices
    are removed, so the shuffled-val / shuffled-test samples don't
    overlap with the sequential blocks regardless of placement.
    """
    remaining = sorted_items.copy()
    test_seq = val_seq = None
    test_list = []
    val_list = []

    # Sequential test (most recent) -- same for both placements.
    if n_test_seq > 0:
        test_seq = remaining[-n_test_seq:]
        test_list.append(test_seq)
        remaining = remaining[:-n_test_seq]

    # Sequential val: newest end (forward) or oldest end (backward).
    if n_val_seq > 0:
        if effective_val_placement == "backward":
            val_seq = remaining[:n_val_seq]
            val_list.append(val_seq)
            remaining = remaining[n_val_seq:]
        else:
            val_seq = remaining[-n_val_seq:]
            val_list.append(val_seq)
            remaining = remaining[:-n_val_seq]

    # Shuffled test from remaining. rng.choice(N, k, replace=False) raises
    # when k > N; that surfaces as a ValueError far from the cause (e.g.
    # when the sequential test block already consumed most of the input,
    # or test_size + val_size jointly exceed n_total). Clamp with a WARN
    # so the caller sees the issue and the split still produces a result.
    if n_test_shuf > 0:
        _eff_test_shuf = min(n_test_shuf, len(remaining))
        if _eff_test_shuf < n_test_shuf:
            logger.warning(
                "Shuffled test size %d exceeds remaining pool %d after "
                "sequential allocation; clamping to %d.",
                n_test_shuf, len(remaining), _eff_test_shuf,
            )
        if _eff_test_shuf > 0:
            test_shuf_idx = rng.choice(len(remaining), _eff_test_shuf, replace=False)
            test_list.append(remaining[test_shuf_idx])
            remaining = np.delete(remaining, test_shuf_idx)

    # Shuffled val from remaining (same clamp rationale).
    if n_val_shuf > 0:
        _eff_val_shuf = min(n_val_shuf, len(remaining))
        if _eff_val_shuf < n_val_shuf:
            logger.warning(
                "Shuffled val size %d exceeds remaining pool %d after "
                "sequential / test allocation; clamping to %d.",
                n_val_shuf, len(remaining), _eff_val_shuf,
            )
        if _eff_val_shuf > 0:
            val_shuf_idx = rng.choice(len(remaining), _eff_val_shuf, replace=False)
            val_list.append(remaining[val_shuf_idx])
            remaining = np.delete(remaining, val_shuf_idx)

    test_items = np.concatenate(test_list) if test_list else np.array([], dtype=remaining.dtype)
    val_items = np.concatenate(val_list) if val_list else np.array([], dtype=remaining.dtype)
    train_items = remaining

    return train_items, val_items, test_items, val_seq, test_seq

def _build_details(timestamps, idx, sequential_idx, n_shuffled, unit) -> str:
    """Build a detail string for a split set.

    Format: ``{first_date}/{last_date} +{n_shuffled}{unit}``

    The ``+N<unit>`` suffix means "N extra items, sampled at random from
    *outside* the sequential date window, mixed into this split." It lets
    the user know at a glance that the split is not purely contiguous.
    ``unit`` is a single letter:
      * ``R`` -- ``N`` additional **rows** (row-based splitting)
      * ``D`` -- ``N`` additional **days** (whole-day splitting)

    Example: ``90_000 val rows 2014-01-20/2014-04-05 +45000R`` =
    45k val rows from outside the Jan-Apr 2014 window were shuffled in
    on top of the sequential 45k that fell inside the window.
    """
    def _fmt_ts(value):
        """Format a single timestamp for log lines. Datetime-typed
        values use ``%Y-%m-%d``; numeric (int/float epoch-seconds or
        generic numeric proxy) values fall back to ``repr(value)``
        because the ``%Y-%m-%d`` format-spec raises
        ``ValueError: Invalid format specifier '%Y-%m-%d' for object
        of type 'int'`` on non-datetime inputs (the FTE's ``ts_field``
        plumbing accepts any monotone numeric column).
        """
        try:
            return format(value, "%Y-%m-%d")
        except (ValueError, TypeError):
            return str(value)

    if sequential_idx is not None and len(sequential_idx) > 0:
        details = f"{_fmt_ts(timestamps.iloc[sequential_idx].min())}/{_fmt_ts(timestamps.iloc[sequential_idx].max())}"
        if n_shuffled > 0:
            details += f" +{n_shuffled}{unit}"
    else:
        if len(idx) > 0:
            details = f"{_fmt_ts(timestamps.iloc[idx].min())}/{_fmt_ts(timestamps.iloc[idx].max())}"
        else:
            details = ""
    return details

# Calculate split sizes
