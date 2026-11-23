"""Cluster 40 regression: the frame content hash depends on row order (audit 2026-11-20, finding 40-F20)."""

import pandas as pd
import pytest

pl = pytest.importorskip("polars")
from mlframe.training.pipeline import _pipeline_cache as pc


def test_polars_frame_hash_differs_when_rows_are_permuted():
    """The same rows in another order must not share a cache key (40-F20)."""
    a = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    b = a[[3, 2, 1, 0]]
    assert pc._hash_polars_frame_rows(a) != pc._hash_polars_frame_rows(b)
    assert pc._hash_polars_frame_rows(a) == pc._hash_polars_frame_rows(a.clone())


def test_pandas_frame_hash_differs_when_rows_are_permuted():
    """The pandas path (via the polars kernel) is order-sensitive too (40-F20)."""
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    b = a.iloc[[3, 2, 1, 0]]
    ha = pc._hash_pandas_frame_via_polars(a)
    hb = pc._hash_pandas_frame_via_polars(b)
    if ha is None or hb is None:
        pytest.skip("pandas-via-polars fastpath unavailable")
    assert ha != hb
