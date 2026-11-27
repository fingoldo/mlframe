"""Cluster 17 regression: the RMSLE and MAPE warn-dedup sets are bounded (audit 2026-11-20, finding 17-F21).

Each distinct (count, total) pair used to add an entry that was never pruned. The sets are now cleared at the bound, so a long run cannot grow
them without limit.
"""

import numpy as np

from mlframe.metrics.regression import _regression_extras as rx


def test_rmsle_warn_set_stays_within_the_bound():
    """Filling the set past the bound and then warning again keeps it at or under the bound (17-F21)."""
    rx._RMSLE_NEG_WARN_SEEN.update((-i - 1, 10_000 + i) for i in range(rx._WARN_SEEN_MAX + 50))
    y = np.array([-1.0, 2.0, 3.0])
    rx.fast_rmsle(y, np.array([1.0, 2.0, 3.0]))
    assert len(rx._RMSLE_NEG_WARN_SEEN) <= rx._WARN_SEEN_MAX
    rx._RMSLE_NEG_WARN_SEEN.clear()


def test_mape_warn_set_stays_within_the_bound():
    """The same bound holds for the MAPE mean-zero warn set (17-F21)."""
    rx._MAPE_MEAN_ZERO_WARN_SEEN.update((i + 1, 10_000 + i) for i in range(rx._WARN_SEEN_MAX + 50))
    y = np.array([0.0, 0.0, 5.0])
    rx.fast_mape_mean(y, np.array([1.0, 1.0, 5.0]))
    assert len(rx._MAPE_MEAN_ZERO_WARN_SEEN) <= rx._WARN_SEEN_MAX
    rx._MAPE_MEAN_ZERO_WARN_SEEN.clear()


def test_a_full_table_starts_over_with_exactly_the_new_key_and_a_table_below_the_bound_keeps_every_key():
    """The shared helper keeps every key while the table has room, and a table past the bound is replaced by just the key being added."""
    seen: set = {(i, i) for i in range(10)}
    rx._remember_warning(seen, (99, 99))
    assert len(seen) == 11 and (99, 99) in seen and (0, 0) in seen
    full: set = {(i, i) for i in range(rx._WARN_SEEN_MAX)}
    rx._remember_warning(full, (-1, -1))
    assert full == {(-1, -1)}
