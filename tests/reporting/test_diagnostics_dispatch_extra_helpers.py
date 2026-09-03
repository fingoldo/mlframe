"""Regression tests for two small helpers in reporting._diagnostics_dispatch_extra.

REPORTING_A-1 / REPORTING_A-2 (2026-08-05 audit):
- _ranked_top_features picked NaN-importance features as top-ranked (np.argsort sorts NaN last
  ascending, first after the prior `[::-1]` reversal).
- _first_group_column's categorical-dtype detector used `dt is object` (identity vs the Python
  builtin `object` TYPE), which a pandas object-dtype column never satisfies, so the branch was
  dead code; there was also no polars dtype branch at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting._diagnostics_dispatch_extra import _first_group_column, _ranked_top_features


def test_ranked_top_features_excludes_nan_importance_from_top():
    """NaN-importance features must not be picked as top-ranked."""
    names = ["a", "b", "c", "d"]
    importances = [np.nan, 0.9, 0.5, 0.1]
    top2 = _ranked_top_features(names, importances, k=2)
    assert "a" not in top2, f"NaN-importance feature 'a' should not rank in the top-2; got {top2}"
    assert top2 == ["b", "c"]


def test_ranked_top_features_all_finite_unaffected():
    """Sanity: with no NaN present, ranking is unchanged (real importances, descending)."""
    names = ["a", "b", "c"]
    importances = [0.1, 0.9, 0.5]
    assert _ranked_top_features(names, importances, k=3) == ["b", "c", "a"]


def test_first_group_column_detects_pandas_object_dtype_column():
    """A plain object-dtype (string) column with bounded cardinality must be detected."""
    df = pd.DataFrame(
        {
            "id_like": [f"row_{i}" for i in range(200)],  # high-cardinality, must be skipped
            "group_col": (["red", "green", "blue"] * 67)[:200],  # object dtype, cardinality 3
        }
    )
    result = _first_group_column(df, ["id_like", "group_col"], max_card=50)
    assert result == "group_col", f"pre-fix: 'dt is object' never matched, so this returned None; got {result!r}"


def test_first_group_column_detects_pandas_category_dtype_column():
    """A pandas category-dtype column with bounded cardinality must still be detected (unaffected by the fix)."""
    df = pd.DataFrame({"cat_col": pd.Categorical(["x", "y"] * 100)})
    result = _first_group_column(df, ["cat_col"], max_card=50)
    assert result == "cat_col"


def test_first_group_column_high_cardinality_object_column_skipped():
    """A high-cardinality (id-like) object column must not be picked."""
    df = pd.DataFrame({"id_like": [f"row_{i}" for i in range(200)]})
    assert _first_group_column(df, ["id_like"], max_card=50) is None


def test_first_group_column_detects_polars_string_column():
    """A polars Utf8/String column with bounded cardinality must be detected (new polars branch)."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"group_col": (["red", "green", "blue"] * 67)[:200]})
    result = _first_group_column(df, ["group_col"], max_card=50)
    assert result == "group_col"
