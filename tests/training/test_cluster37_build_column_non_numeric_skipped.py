"""Cluster 37 regression: a non-numeric split column is skipped, not raised, when the column is rebuilt (audit 2026-11-20, finding 37-F47).

The float64 cast used to sit outside the guarded materialisation, so a string column raised ValueError. The cast is now inside the guard and the
split is skipped; a numeric column is still reassembled.
"""

import numpy as np
import pandas as pd

from mlframe.training.core._misc_helpers import _build_full_column_from_splits


def test_string_column_does_not_raise():
    """A string-valued split column is skipped and the rebuilt column is NaN there (37-F47)."""
    train = pd.DataFrame({"c": ["a", "b"]})
    empty = pd.DataFrame({"c": []})
    out = _build_full_column_from_splits(
        "c",
        train,
        empty,
        empty,
        np.array([0, 1]),
        np.array([], dtype=int),
        np.array([], dtype=int),
        2,
    )
    assert np.isnan(out).all()


def test_numeric_column_is_reassembled():
    """Control: a numeric split column is written at its row indices (37-F47)."""
    train = pd.DataFrame({"c": [1.0, 2.0]})
    empty = pd.DataFrame({"c": []})
    out = _build_full_column_from_splits(
        "c",
        train,
        empty,
        empty,
        np.array([0, 1]),
        np.array([], dtype=int),
        np.array([], dtype=int),
        2,
    )
    np.testing.assert_allclose(out, [1.0, 2.0])
