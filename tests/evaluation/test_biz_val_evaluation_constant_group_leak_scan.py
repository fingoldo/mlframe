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
    import pytest

    df = pd.DataFrame({"col": [1, 2, 3]})
    y = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        constant_group_target_scan(df, y, candidate_cols=["col"])


def test_constant_group_target_scan_no_eligible_groups_not_flagged():
    df = pd.DataFrame({"col": list(range(5))})
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    result = constant_group_target_scan(df, y, candidate_cols=["col"], min_group_size=20)
    assert bool(result.iloc[0]["flagged"]) is False
