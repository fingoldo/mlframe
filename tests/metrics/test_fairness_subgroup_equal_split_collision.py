"""SA24 regression: equal-sized val/test fairness subgroups must not collide.

Pre-fix ``create_fairness_subgroups_indices`` keyed the result dict by
``len(arr)`` ALONE, so when val and test had the same number of rows the second
split silently overwrote the first (only a warning). The partitions are positional
per split and differ when the splits hold different rows, so the overwrite
corrupted whichever split was looked up by length. The fix keys each split's
partition by a stable identity ("train"/"val"/"test"); both equal-sized splits are
now retained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.metrics._fairness_metrics import (
    create_fairness_subgroups,
    create_fairness_subgroups_indices,
)


def _partition_for(bins_series, arr, bin_name):
    """Helper: Partition for."""
    sub = bins_series.loc[arr].to_numpy()
    return np.where(sub == bin_name)[0]


def test_equal_sized_val_test_both_retained_by_name():
    """Equal sized val test both retained by name."""
    n = 600
    rng = np.random.default_rng(0)
    region = pd.Series([f"r{v}" for v in rng.integers(0, 4, n)], name="region")
    sg = create_fairness_subgroups(pd.DataFrame({"region": region}), ["region"], min_pop_cat_thresh=5)

    train_idx = np.arange(n)
    # val and test are DISJOINT but EQUAL-sized (200 each) -> pre-fix collision.
    val_idx = np.arange(200)
    test_idx = np.arange(200, 400)
    assert len(val_idx) == len(test_idx)

    res = create_fairness_subgroups_indices(sg, train_idx, val_idx, test_idx)

    # Both splits retained under stable-identity keys.
    assert "val" in res and "test" in res, "split-name keys must both be present"

    bins_series = sg["region"]["bins"]
    some_bin = bins_series.iloc[val_idx].unique()[0]

    got_val = res["val"]["region"]["bins"][some_bin]
    got_test = res["test"]["region"]["bins"][some_bin]
    exp_val = _partition_for(bins_series, val_idx, some_bin)
    exp_test = _partition_for(bins_series, test_idx, some_bin)

    # The val and test partitions for the SAME bin are different row sets (disjoint
    # splits). Pre-fix res[200] held only the LAST-written split, so retrieving the
    # other split's partition by length gave the wrong rows.
    assert np.array_equal(got_val, exp_val), "val partition corrupted / overwritten"
    assert np.array_equal(got_test, exp_test), "test partition corrupted / overwritten"
    assert not np.array_equal(exp_val, exp_test), "fixture must have distinct val/test partitions"


def test_distinct_sized_splits_still_length_keyed():
    """Backward-compat: when splits have distinct lengths the length keys still
    resolve (the booster eval-metric path relies on len(y_true))."""
    n = 500
    rng = np.random.default_rng(1)
    region = pd.Series([f"r{v}" for v in rng.integers(0, 3, n)], name="region")
    sg = create_fairness_subgroups(pd.DataFrame({"region": region}), ["region"], min_pop_cat_thresh=5)
    res = create_fairness_subgroups_indices(sg, np.arange(n), np.arange(300), np.arange(300, 500))
    assert 300 in res and 200 in res and 500 in res
