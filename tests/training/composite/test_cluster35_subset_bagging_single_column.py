"""Audit 2026-11-20 finding 35-F32: correlation_cluster_feature_subsets accepts a single numeric column.

Before the fix the one-column case ran linkage on an empty condensed matrix and raised ValueError.
"""

import numpy as np
import pandas as pd

from mlframe.training.composite.feature_subset_bagging import correlation_cluster_feature_subsets


def test_single_numeric_column_returns_that_column():
    """One numeric column gives one subset holding it, with no error."""
    X = pd.DataFrame({"a": np.arange(20.0), "label": ["x"] * 20})
    out = correlation_cluster_feature_subsets(X, n_subsets=3, subset_size=1)
    assert out == [["a"]]
