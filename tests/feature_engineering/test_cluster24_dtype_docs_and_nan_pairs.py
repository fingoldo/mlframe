"""Cluster 24 regressions: window targets keep float64, and the variance gate names columns it drops for NaN (audit 2026-11-20, 24-F29, 24-F27)."""

import logging

import numpy as np
import pandas as pd

import importlib

vg = importlib.import_module("mlframe.feature_engineering.variance_gated_pairwise_diff")
from mlframe.feature_engineering.timeseries import create_windowed_features


def test_float64_targets_are_not_rounded_by_the_float32_feature_dtype():
    """A target of 16777217.0 survives the default float32 feature dtype (24-F29)."""
    df = pd.DataFrame({"w": np.ones(30, dtype=np.float64)})

    def past_fcn(df, row_features, targets, features_names, dataset_name):
        """Append one constant feature."""
        row_features.append(1.0)
        features_names.append(dataset_name)

    def future_fcn(df, row_features, targets, features_names, dataset_name):
        """Append one constant feature."""
        row_features.append(1.0)
        features_names.append(dataset_name)

    def targets_fcn(past_windows, future_windows):
        """A float64 target that float32 cannot represent."""
        return 16777217.0

    X, Y = create_windowed_features(
        df=df, start_index=10, end_index=28, step_size=17, past_processing_fcn=past_fcn, future_processing_fcn=future_fcn,
        targets_creation_fcn=targets_fcn, past_windows={"": [1]}, future_windows={"w": [2, 10]},
    )
    assert Y.dtype == np.float64
    assert float(Y.iloc[0]) == 16777217.0
    assert np.asarray(X).dtype == np.float32


def test_nan_columns_are_named_in_a_warning(caplog):
    """One NaN in column a drops every a-b pair and now says so (24-F27)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    df.loc[3, "a"] = np.nan
    with caplog.at_level(logging.WARNING, logger=vg.logger.name):
        vg.variance_gated_pairwise_diff(df, columns=["a", "b", "c"], min_variance=0.0)
    assert any("'a'" in r.getMessage() for r in caplog.records)
