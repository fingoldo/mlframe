"""biz_value test for the multi-split majority-vote mode of ``ks_stability_filter``.

The win: with only ONE train/test KS check, a genuinely stable feature (identical train/test distribution)
still gets falsely flagged unstable at roughly the ``p_value_threshold`` rate purely by sampling noise --
that is what a p-value threshold means. Running SEVERAL independent random subsample splits and requiring a
STRICT MAJORITY of them to fail before flagging drives that false-flag rate down sharply, because a majority
of ``n_splits`` independent low-probability events is far less likely than any single one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._ks_stability import ks_stability_filter


def test_biz_val_ks_stability_filter_multi_split_reduces_false_flag_rate():
    n = 300
    n_seeds = 200
    p_value_threshold = 0.05
    n_splits = 9

    single_false_flags = 0
    multi_false_flags = 0

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        train_df = pd.DataFrame({"stable": rng.normal(0, 1, n)})
        test_df = pd.DataFrame({"stable": rng.normal(0, 1, n)})

        single_result = ks_stability_filter(train_df, test_df, p_value_threshold=p_value_threshold)
        if not bool(single_result.iloc[0]["stable"]):
            single_false_flags += 1

        multi_result = ks_stability_filter(train_df, test_df, p_value_threshold=p_value_threshold, n_splits=n_splits, random_state=seed)
        if not bool(multi_result.iloc[0]["stable"]):
            multi_false_flags += 1

    single_false_flag_rate = single_false_flags / n_seeds
    multi_false_flag_rate = multi_false_flags / n_seeds

    # A single KS check on identical distributions falsely flags at roughly the threshold rate (~5%);
    # majority-vote across n_splits independent subsamples needs a majority of those low-probability
    # events to co-occur, which is far rarer -- assert a real, substantial reduction, not just "<=".
    assert single_false_flag_rate >= 0.02
    assert multi_false_flag_rate <= single_false_flag_rate * 0.6
    assert multi_false_flag_rate <= 0.03


def test_ks_stability_filter_n_splits_default_matches_single_pair_bit_identical():
    rng = np.random.default_rng(1)
    n = 500
    train_df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(2, 1, n)})
    test_df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})

    default_result = ks_stability_filter(train_df, test_df)
    explicit_single_split_result = ks_stability_filter(train_df, test_df, n_splits=1)

    pd.testing.assert_frame_equal(default_result, explicit_single_split_result)
    assert "n_splits" not in default_result.columns
    assert "n_unstable_splits" not in default_result.columns
