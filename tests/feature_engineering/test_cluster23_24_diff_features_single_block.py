"""Cluster 23/24 regressions: entity diffs concat once, and the pairwise diff block is written without a second copy (audit 2026-11-20, findings 23-F7 and 24-F28)."""

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_engineering.entity_diff_features import entity_diff_features
from mlframe.feature_engineering.variance_gated_pairwise_diff import variance_gated_pairwise_diff


def test_wide_entity_diff_is_correct_and_raises_no_performance_warning():
    """300 numeric columns: every diff column equals value minus the entity's previous value, the first row of an entity is NaN, and no PerformanceWarning fires."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(rng.normal(size=(n, 300)), columns=[f"c{i}" for i in range(300)])
    df["entity"] = np.repeat(np.arange(4), 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        out = entity_diff_features(df, "entity")
    assert out.shape[1] == df.shape[1] + 300
    expected = df["c7"] - df.groupby("entity")["c7"].shift(1)
    pd.testing.assert_series_equal(out["c7_diff"].rename("c7"), expected.rename("c7"), check_names=False)
    assert out["c7_diff"].iloc[[0, 10, 20, 30]].isna().all()


def test_a_diff_that_reuses_an_existing_column_name_replaces_it_in_place():
    """A suffix that collides with an existing column overwrites that column, as per-column assignment did."""
    df = pd.DataFrame({"a": [1.0, 3.0, 6.0], "a_diff": [0.0, 0.0, 0.0], "entity": [1, 1, 1]})
    out = entity_diff_features(df, "entity")
    assert list(out.columns).count("a_diff") == 1 and out["a_diff"].tolist()[1:] == [2.0, 3.0]


def test_pairwise_diff_values_names_and_shared_block():
    """Each surviving pair's column is a - b by name, and the result is one float64 block."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": rng.normal(size=200)})
    out = variance_gated_pairwise_diff(df, list(df.columns))
    assert set(out.columns) == {"a__diff__b", "a__diff__c", "b__diff__c"}
    np.testing.assert_allclose(out["a__diff__c"].to_numpy(), (df["a"] - df["c"]).to_numpy())
    assert (out.dtypes == np.float64).all()
