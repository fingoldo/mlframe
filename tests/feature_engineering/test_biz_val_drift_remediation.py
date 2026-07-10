"""biz_value + unit tests for ``feature_engineering.drift_remediation.remediate_drifting_features``.

The win: on a synthetic out-of-time split where one feature's absolute LEVEL drifts with time (train covers
early time_ids, test covers later ones — the raw feature makes train/test trivially separable) while its
within-time_id rank is time-invariant and still carries real signal, the remediation (a) correctly flags the
drifting feature and not a genuinely clean one, and (b) replacing it with its within-group rank measurably
reduces adversarial train/test separability (AUC drops toward 0.5) without discarding the feature outright.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.drift_remediation import remediate_drifting_features
from mlframe.reporting.charts.drift import adversarial_auc


def _make_drift_data(n_time_ids: int, n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    time_ids = np.repeat(np.arange(n_time_ids), n_entities)
    n = time_ids.shape[0]

    # drift_feature's LEVEL grows with time_id (classic Optiver order_count/volume drift): trivially
    # separates an early-time train split from a late-time test split on raw values.
    drift_feature = time_ids.astype(np.float64) * 10.0 + rng.standard_normal(n) * 2.0
    # clean_feature: no time dependence at all -- genuinely non-drifting.
    clean_feature = rng.standard_normal(n)

    df = pd.DataFrame({"time_id": time_ids, "drift_feature": drift_feature, "clean_feature": clean_feature})
    split = n_time_ids // 2
    train_df = df[df["time_id"] < split].reset_index(drop=True)
    test_df = df[df["time_id"] >= split].reset_index(drop=True)
    return train_df, test_df


def test_remediate_drifting_features_flags_the_drifting_column_not_the_clean_one():
    train_df, test_df = _make_drift_data(n_time_ids=60, n_entities=40, seed=0)
    _, _, report = remediate_drifting_features(train_df, test_df, group_col="time_id", n_std=0.5, n_splits=2)

    drift_row = report[report["feature"] == "drift_feature"].iloc[0]
    clean_row = report[report["feature"] == "clean_feature"].iloc[0]
    assert bool(drift_row["flagged"]) is True
    assert bool(clean_row["flagged"]) is False
    assert drift_row["drift_importance"] > clean_row["drift_importance"]


def test_remediate_drifting_features_replaces_flagged_column_with_bounded_rank():
    train_df, test_df = _make_drift_data(n_time_ids=60, n_entities=40, seed=1)
    train_out, test_out, report = remediate_drifting_features(train_df, test_df, group_col="time_id", n_std=0.5, n_splits=2, rank_pct=True)

    assert report.loc[report["feature"] == "drift_feature", "flagged"].iloc[0]
    # rank_pct=True -> normalised [0, 1] ranks, unlike the original unbounded level-drifting values.
    assert train_out["drift_feature"].between(0.0, 1.0).all()
    assert test_out["drift_feature"].between(0.0, 1.0).all()
    # the unflagged clean feature must pass through untouched.
    assert np.array_equal(train_out["clean_feature"].to_numpy(), train_df["clean_feature"].to_numpy())


def test_remediate_drifting_features_missing_group_col_raises():
    train_df, test_df = _make_drift_data(n_time_ids=10, n_entities=5, seed=2)
    with pytest.raises(ValueError):
        remediate_drifting_features(train_df, test_df, group_col="not_a_real_column")


def test_biz_val_remediation_reduces_adversarial_separability():
    train_df, test_df = _make_drift_data(n_time_ids=80, n_entities=50, seed=42)
    feature_cols = ["drift_feature", "clean_feature"]

    auc_before, *_ = adversarial_auc(train_df[feature_cols], test_df[feature_cols], feature_names=feature_cols, n_splits=3, seed=0)

    train_out, test_out, report = remediate_drifting_features(train_df, test_df, group_col="time_id", n_std=0.5, n_splits=3, seed=0)
    assert report.loc[report["feature"] == "drift_feature", "flagged"].iloc[0]

    auc_after, *_ = adversarial_auc(train_out[feature_cols], test_out[feature_cols], feature_names=feature_cols, n_splits=3, seed=0)

    # Raw drift_feature makes train/test trivially separable (level drift with time); floor set well below
    # the measured value (~0.99) to tolerate seed noise while still catching a broken/no-op remediation.
    assert auc_before > 0.90, f"sanity: raw drift_feature should make train/test near-perfectly separable, got AUC={auc_before:.3f}"
    # After remediation the level drift is gone (within-time_id rank only); separability should collapse
    # substantially toward the AUC~0.5 "same distribution" baseline.
    assert auc_after < auc_before - 0.15, (
        f"remediation should measurably reduce adversarial separability: before={auc_before:.3f} after={auc_after:.3f}"
    )
