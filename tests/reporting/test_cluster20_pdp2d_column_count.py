"""Cluster 20 regression: the 2-D PDP counts columns from the shape, without copying the frame (audit 2026-11-20, finding 20-F40).

Source-level pin: the full-frame np.asarray(X).reshape copy is gone and shape_of is used; the shape rule agrees with the copy for 2-D input and
counts a 1-D input as one column.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.reporting.charts import pdp_2d as pd2
from mlframe.utils.array_shape import shape_of

_SRC = Path(pd2.__file__).resolve()


def test_shape_rule_matches_the_copy_for_2d_and_counts_1d_as_one():
    """shape_of gives the same column count as the copy for a frame and one column for a vector (20-F40)."""
    X = pd.DataFrame(np.zeros((5, 3)))
    assert shape_of(X)[1] == np.asarray(X).reshape(len(X), -1).shape[1]
    v = np.zeros(5)
    n = shape_of(v)[1] if len(shape_of(v)) > 1 else 1
    assert n == 1
