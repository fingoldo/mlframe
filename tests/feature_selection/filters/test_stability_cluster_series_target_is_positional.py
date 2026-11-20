"""A Series target with a non-default index is subsampled by position, not by label, in both stability-selection helpers."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._stability_cluster import cluster_stability_selection, complementary_pairs_stability


def _run(fn, **kw):
    """Run ``fn`` on a frame with a Series target indexed by shuffled labels; return the (X_sub col0, y_sub) pairs the selector saw."""
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": rng.standard_normal(n)})
    y = pd.Series(np.arange(n) * 10.0, index=rng.permutation(n) + 1000)
    seen = []

    def sel(Xs, ys):
        """Record the aligned subsample and select feature 0."""
        seen.append((np.asarray(Xs["a"]), np.asarray(ys)))
        return np.array([0])

    fn(X, y, sel, **kw)
    return seen


def test_cluster_stability_subsets_a_series_target_by_position():
    """The target rows handed to the selector are those of the sampled feature rows."""
    seen = _run(cluster_stability_selection, n_bootstrap=3)
    assert seen
    for xa, ys in seen:
        assert np.allclose(ys, xa * 10.0)


def test_complementary_pairs_subsets_a_series_target_by_position():
    """The same alignment holds for the complementary-pairs variant."""
    seen = _run(complementary_pairs_stability, n_pairs=3)
    assert seen
    for xa, ys in seen:
        assert np.allclose(ys, xa * 10.0)
