"""biz_value test for ``feature_selection.filters.monotonic_deviation_stability_filter``.

The win: a feature whose deviation-from-own-group-baseline genuinely and consistently predicts the target
sign across every group subsample should be flagged stable, while a feature with no real relationship (pure
noise, so its apparent sign under any one subsample is a coin flip) should be flagged jumpy/unstable -- this
is the resampling-robustness check the source writeup used to discard features that "looked informative" on
one pass but broke down across different entity baskets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._monotonic_stability import monotonic_deviation_stability_filter


def _make_data(seed: int):
    rng = np.random.default_rng(seed)
    n_groups = 250
    rows_per_group = 6
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(group_ids)

    group_baseline = rng.normal(0, 1, n_groups)[group_ids]
    good_deviation_noise = rng.normal(0, 0.3, n)
    good_feature = group_baseline + good_deviation_noise
    # y depends consistently on the sign of the feature's deviation from ITS OWN group mean.
    y = (good_deviation_noise > 0).astype(int)

    jumpy_feature = rng.normal(0, 1, n)  # no relationship to y whatsoever

    df = pd.DataFrame({"group": group_ids, "good_feature": good_feature, "jumpy_feature": jumpy_feature})
    return df, y


def test_biz_val_monotonic_stability_filter_separates_stable_from_jumpy_feature():
    df, y = _make_data(seed=0)

    result = monotonic_deviation_stability_filter(
        df, y, group_col="group", feature_cols=["good_feature", "jumpy_feature"], n_subsamples=40, random_state=0
    )

    by_name = {row["feature"]: row for _, row in result.iterrows()}
    assert by_name["good_feature"]["stable"], by_name["good_feature"]
    assert not by_name["jumpy_feature"]["stable"], by_name["jumpy_feature"]
    assert by_name["good_feature"]["sign_agreement_fraction"] > by_name["jumpy_feature"]["sign_agreement_fraction"]


def test_monotonic_stability_filter_returns_expected_columns():
    df, y = _make_data(seed=1)
    result = monotonic_deviation_stability_filter(df, y, group_col="group", feature_cols=["good_feature"], n_subsamples=10, random_state=1)
    assert {"feature", "full_sample_correlation", "sign_agreement_fraction", "stable"} <= set(result.columns)
    assert len(result) == 1
