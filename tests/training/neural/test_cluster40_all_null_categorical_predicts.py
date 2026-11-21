"""Regression tests for the audit findings in this module."""
import pytest
"""Audit 2026-11-20 finding 40-F9: an all-null categorical column at fit does not crash predict; every row maps to the unknown code."""

import numpy as np
import pandas as pd

pytest.importorskip("lightning")
from mlframe.training.neural.base._base_fit_prep import _FitPrepMixin


def test_all_null_fit_column_maps_every_predict_row_to_the_unknown_code():
    """A categorical that was all-null at fit has no codes; predict maps its values to the reserved code instead of indexing an empty array."""
    obj = _FitPrepMixin()
    obj._cat_code_maps_ = {"c": {}}
    obj._cat_cols_ = ["c"]
    obj._cat_cardinalities_ = [0]
    obj._input_columns_ = ["c", "x"]
    X = pd.DataFrame({"c": ["a", "b", None], "x": [1.0, 2.0, 3.0]})
    out = obj._apply_cat_codes(X)
    assert (np.asarray(out["c"], dtype=np.float64) == 0.0).all()
