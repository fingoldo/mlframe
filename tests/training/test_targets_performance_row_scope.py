"""The targets quality frame says how many labelled rows a target's metrics come from, without treating it as a metric."""

from __future__ import annotations

import types

import numpy as np

from mlframe.training.targets_performance import ROW_SCOPE_COLUMNS, compare_targets_performance, targets_performance_frame


def _models(rmse_full: float, rmse_part: float):
    entry = lambda rmse: types.SimpleNamespace(model_name="lgb", metrics={"test": {"RMSE": rmse, "MAE": rmse / 2}})
    return {"regression": {"y_full": [entry(rmse_full)], "y_part": [entry(rmse_part)]}}


METADATA = {"target_rows": {"regression/y_part": {"n_labelled": {"test_idx": 12}, "low_n": True}}}


def test_a_target_with_gaps_carries_its_labelled_count_and_low_n():
    frame = targets_performance_frame(_models(1.0, 2.0), METADATA)
    part = frame[frame["target_name"] == "y_part"].iloc[0]
    full = frame[frame["target_name"] == "y_full"].iloc[0]
    assert part["labelled_rows"] == 12 and bool(part["low_n"]) is True
    assert np.isnan(full["labelled_rows"])
    assert "RMSE" in frame.columns


def test_without_targets_with_gaps_the_frame_has_no_row_scope_columns():
    frame = targets_performance_frame(_models(1.0, 2.0), {})
    assert not set(ROW_SCOPE_COLUMNS) & set(frame.columns)


def test_labelled_counts_are_not_compared_as_metrics():
    runs = {"a": targets_performance_frame(_models(1.0, 2.0), METADATA), "b": targets_performance_frame(_models(1.1, 2.1), METADATA)}
    comparison = compare_targets_performance(runs)
    text = comparison.frame.to_string() + comparison.scores.to_string()
    assert "labelled_rows" not in text and "low_n" not in text
    assert "RMSE" in text
