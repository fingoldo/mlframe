"""The BLAS correlation of the meta scorer equals pandas' ``corr``."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import _inter_x_max_corr, _pearson_matrix


def test_matrix_equals_pandas_corr():
    """Finite frames agree to rounding (with a constant column giving NaN in both); a frame with NaN takes the pandas path."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(4000, 12)), columns=[f"c{i}" for i in range(12)])
    X["c3"] = X["c0"] * 0.9 + X["c1"] * 0.1
    X["const"] = 4.0
    want = X.corr(method="pearson").to_numpy()
    got = _pearson_matrix(X)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)
    X.loc[::17, "c5"] = np.nan
    np.testing.assert_array_equal(_pearson_matrix(X), X.corr(method="pearson").to_numpy())
    assert _inter_x_max_corr(X.drop(columns=["const"])) > 0.9
