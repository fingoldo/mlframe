"""Cluster 25 regression: the ICE metric builds its class indicator in one helper for numpy, pandas and polars inputs (audit 2026-11-20, finding 25-F50)."""

import numpy as np
import pandas as pd
import polars as pl

from mlframe.metrics._ice_metric import _class_indicator_raw


def test_indicator_for_multiclass_labels_and_multilabel_columns():
    """Multiclass: equality with the class id or with ``labels[id]``; multilabel: the column; numpy, pandas and polars all work."""
    y = np.array([0, 2, 1, 2])
    np.testing.assert_array_equal(_class_indicator_raw(y, 2, None, False), [False, True, False, True])
    np.testing.assert_array_equal(_class_indicator_raw(np.array(["a", "c", "b", "c"]), 1, ["a", "c", "b"], False), [False, True, False, True])
    assert _class_indicator_raw(pd.Series(y), 1, None, False).tolist() == [False, False, True, False]
    assert _class_indicator_raw(pl.Series(y), 0, None, False).to_list() == [True, False, False, False]
    ml = np.array([[1, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(_class_indicator_raw(ml, 1, None, True), [0, 1, 1])
