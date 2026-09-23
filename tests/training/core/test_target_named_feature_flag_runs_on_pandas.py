"""The name-based leakage flag must run on a pandas frame, whose ``.columns`` raises on a truth test."""

import pandas as pd

from mlframe.training.core._main_train_suite_target_distribution import _flag_target_named_features


def test_a_pandas_frame_is_flagged_not_skipped():
    df = pd.DataFrame({"target_outcome_amount": [1.0, 2.0], "x": [3.0, 4.0]})
    metadata: dict = {}
    _flag_target_named_features(df, {"regression": {"target_outcome": None}}, metadata)
    assert metadata["target_named_feature_columns"] == ["target_outcome_amount"]


def test_a_frame_without_columns_is_a_no_op():
    metadata: dict = {}
    _flag_target_named_features(object(), {"regression": {"t": None}}, metadata)
    assert metadata == {}
