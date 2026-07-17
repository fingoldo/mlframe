"""biz_value test for ``evaluation.constant_group_target_scan``.

The win: reproduces the exact source scenario (target near-constant when grouped by application date) and
confirms the leaky column is flagged, while a genuinely non-leaky grouping column (target variance matches
the overall pool within each group) is correctly not flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan


def test_biz_val_constant_group_target_scan_flags_the_leaky_column():
    """Constant group target scan flags the leaky column."""
    rng = np.random.default_rng(0)
    n_dates = 30
    rows_per_date = 50
    n = n_dates * rows_per_date

    application_date = np.repeat(np.arange(n_dates), rows_per_date)
    # the target is a near-deterministic function of application_date (a data-generation quirk), not of
    # any genuine feature -- tiny noise so it's not EXACTLY constant (realistic).
    date_rate = rng.uniform(0.05, 0.95, n_dates)
    y_leaky = (rng.random(n) < date_rate[application_date]).astype(float)
    # force each date-group to be near-deterministic (either mostly-0 or mostly-1) to mimic the reported quirk.
    y_leaky = np.where(date_rate[application_date] > 0.5, 1.0, 0.0)
    flip_mask = rng.random(n) < 0.02
    y_leaky = np.where(flip_mask, 1 - y_leaky, y_leaky)

    non_leaky_group = rng.integers(0, n_dates, n)  # unrelated grouping key

    df = pd.DataFrame({"application_date": application_date, "random_group": non_leaky_group})
    result = constant_group_target_scan(df, y_leaky, candidate_cols=["application_date", "random_group"], min_group_size=20)

    by_col = {row["column"]: row for _, row in result.iterrows()}
    assert by_col["application_date"]["flagged"] is True, by_col["application_date"]
    assert by_col["random_group"]["flagged"] is False, by_col["random_group"]
    assert by_col["application_date"]["min_group_variance_ratio"] < by_col["random_group"]["min_group_variance_ratio"]


def test_constant_group_target_scan_zero_overall_variance_raises():
    """Constant group target scan zero overall variance raises."""
    import pytest

    df = pd.DataFrame({"col": [1, 2, 3]})
    y = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        constant_group_target_scan(df, y, candidate_cols=["col"])


def test_constant_group_target_scan_no_eligible_groups_not_flagged():
    """Constant group target scan no eligible groups not flagged."""
    df = pd.DataFrame({"col": list(range(5))})
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    result = constant_group_target_scan(df, y, candidate_cols=["col"], min_group_size=20)
    assert bool(result.iloc[0]["flagged"]) is False


def test_biz_val_constant_group_target_scan_combo_finds_compound_key_leak_invisible_alone():
    # a leak that only shows up under the COMPOUND key (branch, weekday): the target rate is near-deterministic
    # per (branch, weekday) pair, but each column marginalized alone washes the pattern out completely, since
    # the per-branch and per-weekday deterministic rates are constructed to average out to ~0.5 individually.
    """Constant group target scan combo finds compound key leak invisible alone."""
    rng = np.random.default_rng(1)
    n_branches = 6
    n_weekdays = 7
    rows_per_combo = 40
    branches = np.repeat(np.arange(n_branches), n_weekdays * rows_per_combo)
    weekdays = np.tile(np.repeat(np.arange(n_weekdays), rows_per_combo), n_branches)
    n = branches.shape[0]

    # deterministic rate per (branch, weekday) pair alternates 0/1 in a checkerboard so branch-marginal and
    # weekday-marginal rates both average to ~0.5 -- no single column shows a near-constant group.
    combo_rate = ((branches + weekdays) % 2).astype(float)
    flip_mask = rng.random(n) < 0.02
    y_combo_leak = np.where(flip_mask, 1 - combo_rate, combo_rate)

    df = pd.DataFrame({"branch": branches, "weekday": weekdays})

    single = constant_group_target_scan(df, y_combo_leak, candidate_cols=["branch", "weekday"], min_group_size=20)
    by_col = {row["column"]: row for _, row in single.iterrows()}
    assert by_col["branch"]["flagged"] is False, by_col["branch"]
    assert by_col["weekday"]["flagged"] is False, by_col["weekday"]

    combo = constant_group_target_scan(df, y_combo_leak, candidate_cols=["branch", "weekday"], min_group_size=20, combo_max_size=2)
    by_key = {row["column"]: row for _, row in combo.iterrows()}
    combo_row = by_key[("branch", "weekday")]
    assert combo_row["flagged"] is True, combo_row
    assert combo_row["min_group_variance_ratio"] < 0.1
    # the compound key is far more deterministic than either marginal column alone.
    assert combo_row["min_group_variance_ratio"] < by_key["branch"]["min_group_variance_ratio"]
    assert combo_row["min_group_variance_ratio"] < by_key["weekday"]["min_group_variance_ratio"]


def test_constant_group_target_scan_combo_max_size_one_is_bit_identical_to_default():
    # combo_max_size=1 (the implicit default) must reproduce the exact original single-column-only output --
    # the multi-column mode is strictly opt-in.
    """Constant group target scan combo max size one is bit identical to default."""
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"a": rng.integers(0, 10, n), "b": rng.integers(0, 5, n)})
    y = rng.random(n)

    baseline = constant_group_target_scan(df, y, candidate_cols=["a", "b"], min_group_size=10)
    explicit = constant_group_target_scan(df, y, candidate_cols=["a", "b"], min_group_size=10, combo_max_size=1)
    pd.testing.assert_frame_equal(baseline, explicit)


def test_constant_group_target_scan_combo_max_cols_bounds_combination_count():
    """Constant group target scan combo max cols bounds combination count."""
    rng = np.random.default_rng(3)
    n = 300
    cols = {f"c{i}": rng.integers(0, 4, n) for i in range(6)}
    df = pd.DataFrame(cols)
    y = rng.random(n)

    result = constant_group_target_scan(df, y, candidate_cols=list(cols.keys()), min_group_size=10, combo_max_size=2, combo_max_cols=3)
    combo_rows = [row for row in result["column"] if isinstance(row, tuple)]
    # C(3, 2) = 3 combinations from the first 3 columns only, not C(6, 2) = 15.
    assert len(combo_rows) == 3
