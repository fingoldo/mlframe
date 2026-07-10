"""biz_value + unit tests for ``evaluation.subpopulation_drift.subpopulation_ratio_drift_check``.

The win: on a synthetic mirroring the home-credit 5th-place diagnosis (majority subgroup ~90/10 in train vs
~99/1 in test — a real ~10x prevalence-ratio shift on the minority subgroup), the check (a) correctly flags
the shifted minority subgroup and not a genuinely stable subgroup, and (b) recovers the TRUE generative
prevalence ratio to within sampling tolerance, proving it's a meaningfully quantitative early-warning signal
for "a subgroup-only feature built on this column is at overfit risk," not just a boolean smoke test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check


def _make_shifted_subpopulation_data(n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    train_shift_col = rng.choice(["A", "B"], size=n_train, p=[0.90, 0.10])
    test_shift_col = rng.choice(["A", "B"], size=n_test, p=[0.99, 0.01])
    # a genuinely stable column: same 50/50 split in both train and test.
    train_stable_col = rng.choice(["X", "Y"], size=n_train, p=[0.5, 0.5])
    test_stable_col = rng.choice(["X", "Y"], size=n_test, p=[0.5, 0.5])
    train_df = pd.DataFrame({"loan_type": train_shift_col, "region": train_stable_col})
    test_df = pd.DataFrame({"loan_type": test_shift_col, "region": test_stable_col})
    return train_df, test_df


def test_subpopulation_ratio_drift_check_flags_shifted_minority_subgroup():
    train_df, test_df = _make_shifted_subpopulation_data(50_000, 50_000, seed=0)
    report = subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="loan_type", ratio_threshold=2.0)

    row_b = report[report["subgroup_value"] == "B"].iloc[0]
    row_a = report[report["subgroup_value"] == "A"].iloc[0]
    assert bool(row_b["flagged"]) is True
    assert bool(row_a["flagged"]) is False


def test_subpopulation_ratio_drift_check_does_not_flag_stable_column():
    train_df, test_df = _make_shifted_subpopulation_data(20_000, 20_000, seed=1)
    report = subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="region", ratio_threshold=2.0)
    assert not report["flagged"].any()


def test_subpopulation_ratio_drift_check_value_only_in_one_split_is_inf_ratio():
    train_df = pd.DataFrame({"col": ["A"] * 10 + ["B"] * 5})
    test_df = pd.DataFrame({"col": ["A"] * 10})
    report = subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="col")
    row_b = report[report["subgroup_value"] == "B"].iloc[0]
    assert row_b["prevalence_ratio"] == float("inf")
    assert bool(row_b["flagged"]) is True


def test_subpopulation_ratio_drift_check_missing_column_raises():
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    test_df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="not_a_column")


def test_biz_val_subpopulation_ratio_drift_check_recovers_true_generative_ratio():
    train_df, test_df = _make_shifted_subpopulation_data(200_000, 200_000, seed=42)
    report = subpopulation_ratio_drift_check(train_df, test_df, subgroup_col="loan_type")

    row_b = report[report["subgroup_value"] == "B"].iloc[0]
    # True generative ratio: train_B=0.10, test_B=0.01 -> ratio = 10.0. At n=200k per split the sampling
    # noise on a 1%/10% proportion estimate is small; assert recovery within 15% relative tolerance
    # (measured deviation is typically <5%) -- a meaningfully accurate quantitative estimate, not a coin flip.
    assert row_b["prevalence_ratio"] == pytest.approx(10.0, rel=0.15)
