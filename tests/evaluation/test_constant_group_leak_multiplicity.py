"""A grouping column must not be called a leak because its MINIMUM over many groups is small.

The scan flagged on `min(group variance) / overall variance < 0.1` with no adjustment for how many groups were
searched - `n_groups` was computed and reported but never entered the decision. With 4000 groups of five rows and a
pure-noise target, one group reaches a ratio of 0.02 by chance alone, and the report called it "a de facto leak ...
suspiciously more deterministic than chance".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan


def test_a_random_partition_with_many_groups_is_not_flagged():
    """The minimum variance ratio over thousands of random groups is small by construction, and must not read as a leak."""
    rng = np.random.default_rng(3)
    n_groups, size = 4000, 5
    df = pd.DataFrame({"g": np.repeat(np.arange(n_groups), size)})
    y = rng.normal(size=n_groups * size)
    row = constant_group_target_scan(df, y, ["g"], min_group_size=5).iloc[0]
    assert row["min_group_variance_ratio"] < 0.1, "this bed only means something while the raw ratio clears the bar"
    assert row["min_group_variance_pvalue"] > 0.05
    assert not row["flagged"], "a minimum over 4000 groups is not evidence of a leak"


def test_a_real_constant_group_leak_is_still_flagged():
    """The correction must not cost the detection it exists to qualify: a target that IS the group still trips the scan."""
    rng = np.random.default_rng(0)
    n = 16_000
    groups = rng.integers(0, 20, n)
    y = groups.astype(float) + rng.normal(0, 0.01, n)  # the target is a function of the group
    row = constant_group_target_scan(pd.DataFrame({"g": groups}), y, ["g"]).iloc[0]
    assert row["flagged"]
    assert row["min_group_variance_pvalue"] < 1e-6


def test_the_number_of_groups_searched_is_reported():
    """The eligible-group count is what the multiplicity correction is applied over, so the report has to carry it."""
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({"g": rng.integers(0, 50, n)})
    row = constant_group_target_scan(df, rng.normal(size=n), ["g"]).iloc[0]
    assert row["n_eligible_groups"] > 0
    assert row["n_eligible_groups"] <= row["n_groups"]


def test_a_column_with_no_eligible_group_reports_no_verdict():
    """Nothing large enough to test means NaN and unflagged, not a verdict of clean."""
    df = pd.DataFrame({"g": np.arange(50)})  # every group has one row
    row = constant_group_target_scan(df, np.random.default_rng(0).normal(size=50), ["g"]).iloc[0]
    assert row["n_eligible_groups"] == 0
    assert np.isnan(row["min_group_variance_pvalue"])
    assert not row["flagged"]


def test_family_alpha_is_tunable():
    """A caller who wants the old, uncorrected behaviour can ask for it explicitly."""
    rng = np.random.default_rng(3)
    n_groups, size = 4000, 5
    df = pd.DataFrame({"g": np.repeat(np.arange(n_groups), size)})
    y = rng.normal(size=n_groups * size)
    row = constant_group_target_scan(df, y, ["g"], min_group_size=5, family_alpha=1.0).iloc[0]
    assert row["flagged"], "family_alpha=1 restores the pure-ratio verdict"
