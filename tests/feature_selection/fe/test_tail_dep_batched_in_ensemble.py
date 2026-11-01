"""The batched tail-dependence scorer equals the per-column scorer it replaced in the ensemble."""

import numpy as np

from mlframe.feature_selection.filters._orth_auto_scorer_fe import _score_tail_dep
from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import _batch_one_scorer


def _call(name, x, y):
    """The per-column dispatcher, tail-dependence only."""
    assert name == "tail_dep"
    return _score_tail_dep(x, y, random_state=0)


def test_batch_equals_per_column():
    """Finite columns agree to rounding; a column with a NaN falls back to the per-column scorer (0.0)."""
    rng = np.random.default_rng(6)
    n = 6000
    y = rng.normal(size=n)
    X = rng.normal(size=(n, 9)).astype(np.float32)
    X[:, 1] = (0.8 * y + 0.4 * rng.normal(size=n)).astype(np.float32)  # co-moves with y in both tails
    X[:, 2] = np.round(X[:, 2], 1)  # ties
    kw = dict(nbins=10, random_state=0, n_neighbors=3, copula_n_bins=20, dcor_n_sample=1500, call=_call)
    got = _batch_one_scorer("tail_dep", X, y, **kw)
    want = np.array([_call("tail_dep", X[:, j], y) for j in range(X.shape[1])])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
    assert got.max() > 0
    X2 = X.copy()
    X2[5, 4] = np.nan
    got2 = _batch_one_scorer("tail_dep", X2, y, **kw)
    assert got2[4] == 0.0 and np.allclose(np.delete(got2, 4), np.delete(want, 4), atol=1e-9)
