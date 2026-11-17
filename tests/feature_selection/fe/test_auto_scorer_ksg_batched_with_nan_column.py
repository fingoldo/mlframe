"""The auto-scorer's KSG batch keeps the complete columns in one call when another column has a missing value, with the per-column values."""

import numpy as np

from mlframe.feature_selection.filters import _ksg_cd_fast
from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import _batch_one_scorer, _dispatch_scorer


def _call(name, x, yy):
    """Per-column dispatcher bound to the test's hyper-parameters."""
    return _dispatch_scorer(name, x, yy, 0, nbins=8, n_neighbors=3, copula_n_bins=8, dcor_n_sample=300)


def _batch(X, y):
    """KSG through the batch entry point with the per-column dispatcher as the fallback."""
    return _batch_one_scorer("ksg", X, y, nbins=8, random_state=0, n_neighbors=3, copula_n_bins=8, dcor_n_sample=300, call=_call)


def test_complete_columns_stay_batched_and_equal_the_per_column_values(monkeypatch):
    """One NaN among six columns: five columns are scored by a single kernel entry, the sixth on its finite rows, and every value is the per-column one."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 3] + rng.normal(size=n) > 0).astype(np.int64)
    X[7, 2] = np.nan
    want = np.array([_call("ksg", X[:, j], y) for j in range(6)])

    widths = []
    real = _ksg_cd_fast._mi_cd_columns_auto

    def spy(Xs, d, n_classes, n_neighbors):
        """Record the width of each kernel entry."""
        widths.append(int(Xs.shape[1]))
        return real(Xs, d, n_classes, n_neighbors)

    monkeypatch.setattr(_ksg_cd_fast, "_mi_cd_columns_auto", spy)
    got = _batch(X, y)
    assert sorted(widths) == [1, 5]
    np.testing.assert_array_equal(got, want)
    assert got[0] > got[1] and got[0] > got[4]  # the informative column scores above the noise columns
