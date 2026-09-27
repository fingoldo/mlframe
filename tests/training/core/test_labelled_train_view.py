"""The opt-in diagnostics get the train rows that carry a label, with their group ids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._main_train_suite_phases import _labelled_train_view


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_rows_without_a_label_leave_with_their_features_and_group_ids(backend):
    frame = pd.DataFrame({"x": np.arange(6.0)})
    frame = pl.from_pandas(frame) if backend == "polars" else frame
    y = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    groups = np.array([10, 11, 12, 13, 14, 15])
    train, y_out, g_out = _labelled_train_view(frame, y, groups)
    assert list(np.asarray(train["x"])) == [0.0, 2.0, 3.0, 5.0]
    assert y_out.tolist() == [1.0, 2.0, 3.0, 4.0] and g_out.tolist() == [10, 12, 13, 15]


def test_full_length_group_ids_and_fully_labelled_targets_pass_through():
    frame = pd.DataFrame({"x": np.arange(3.0)})
    full_groups = np.arange(10)
    train, y_out, g_out = _labelled_train_view(frame, np.array([1.0, np.nan, 2.0]), full_groups)
    assert len(train) == 2 and g_out is full_groups
    same, _, _ = _labelled_train_view(frame, np.array([1.0, 2.0, 3.0]), full_groups)
    assert same is frame
