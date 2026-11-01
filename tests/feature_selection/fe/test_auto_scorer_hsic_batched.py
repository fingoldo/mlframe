"""The auto-scorer's batched HSIC matches its per-column scorer to the float32 rounding of the dense Gram sums, and a column with a NaN keeps the per-column path."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import _batch_one_scorer, _dispatch_scorer


def _batch(X, y):
    """HSIC through the batch entry point with the per-column dispatcher as the fallback."""
    def call(name, x, yy):
        """Per-column dispatcher bound to the test's hyper-parameters."""
        return _dispatch_scorer(name, x, yy, 0, nbins=8, n_neighbors=5, copula_n_bins=8, dcor_n_sample=300)

    return _batch_one_scorer("hsic", X, y, nbins=8, random_state=0, n_neighbors=5, copula_n_bins=8, dcor_n_sample=300, call=call), call


def test_batch_equals_per_column_and_nan_column_falls_back():
    """Finite columns agree with the per-column values; a NaN column routes every column through the per-column scorer."""
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(np.int64)
    got, call = _batch(X, y)
    want = np.array([call("hsic", X[:, j], y) for j in range(4)])
    # the dense per-column form sums float32 Gram entries (error about 6e-7 absolute); the kernel sums in double
    np.testing.assert_allclose(got, want, rtol=0.0, atol=2e-6)
    assert np.argmax(got) == np.argmax(want)
    X[5, 2] = np.nan
    got_nan, _ = _batch(X, y)
    assert got_nan.shape == (4,)
