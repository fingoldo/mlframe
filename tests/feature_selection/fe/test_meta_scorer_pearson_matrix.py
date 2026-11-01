"""The BLAS correlation of the meta scorer equals pandas' ``corr``."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import _abs_pearson_matrix, _inter_x_max_corr


def test_matrix_equals_pandas_corr():
    """Finite frames agree to rounding (with a constant column giving NaN in both); a frame with NaN takes the pandas path."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(4000, 12)), columns=[f"c{i}" for i in range(12)])
    X["c3"] = X["c0"] * 0.9 + X["c1"] * 0.1
    X["const"] = 4.0
    want = np.abs(X.corr(method="pearson").to_numpy())
    got = _abs_pearson_matrix(X)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)
    X.loc[::17, "c5"] = np.nan
    X["c5"] = X["c5"].fillna(X["c5"].mean()).where(np.arange(len(X)) % 17 != 0, np.nan)
    with_nan = _abs_pearson_matrix(X)
    off = ~np.eye(with_nan.shape[0], dtype=bool)
    ok = np.isfinite(with_nan[off])
    ref = np.abs(X.corr(method="pearson").to_numpy())[off]
    np.testing.assert_allclose(with_nan[off][ok], ref[ok], rtol=0, atol=1e-9)
    assert _inter_x_max_corr(X.drop(columns=["const"])) > 0.9
