"""Cluster 32 regression: the two TF-IDF feed arms vectorise the same tokens, including nulls (audit 2026-11-20, finding 32-F19)."""

import numpy as np
import pandas as pd

from mlframe.training._benchmarks import bench_tfidf_input_path as bench


def test_both_arms_produce_identical_matrices_with_nulls():
    """Arm A (fillna then astype) and arm B (native NA fill) produce the same TF-IDF matrix (32-F19)."""
    s = pd.Series(["alpha beta", None, "beta gamma", None, "alpha"], dtype=object)
    a = bench._path_values(s)
    b = bench._path_to_numpy(s)
    np.testing.assert_allclose(a.toarray(), b.toarray())
