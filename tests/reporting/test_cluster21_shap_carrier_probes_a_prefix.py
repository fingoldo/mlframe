"""Cluster 21 regression: the SHAP carrier probes a bounded prefix of object columns and skips the whole-frame float copy (audit 2026-11-20, finding 21-F14)."""

import numpy as np
import pandas as pd

from mlframe.reporting.charts import shap_panels as sp


def _frame(n=3000):
    """A numeric column, a string column, and a column of per-row vectors."""
    return pd.DataFrame({"x": np.arange(n, dtype=float), "s": np.where(np.arange(n) % 2 == 0, "a", "b"), "emb": [np.arange(3.0)] * n})


def test_strings_become_categories_and_vector_columns_stay_object():
    """The string column is cast to 'category'; the embedding column is left alone."""
    out = sp._carrier_with_categoricals(_frame())
    assert isinstance(out["s"].dtype, pd.CategoricalDtype)
    assert out["emb"].dtype == object


def test_a_non_scalar_value_beyond_the_probed_prefix_does_not_raise():
    """A list hidden past the probe window makes the category cast fail; the column then stays object instead of raising."""
    n = sp._NONSCALAR_PROBE_ROWS + 500
    col = ["a"] * n
    col[-1] = [1, 2]
    out = sp._carrier_with_categoricals(pd.DataFrame({"x": np.zeros(n), "late": col}))
    assert out["late"].dtype == object


def test_carrier_and_names_skip_the_float_matrix_and_as_frame_still_returns_it():
    """_carrier_and_names gives the carrier and names only; _as_frame_and_names still returns the float matrix as before."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    carrier, names = sp._carrier_and_names(df, None)
    assert names == ["a", "b"] and carrier.shape == (2, 2)
    _carrier2, vals, names2 = sp._as_frame_and_names(df, None)
    assert vals.shape == (2, 2) and names2 == names
