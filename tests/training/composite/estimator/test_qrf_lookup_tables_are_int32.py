"""The quantile forest stores its per-tree row and offset tables as int32, which halves what a fitted model keeps.

At 200 trees and 2M rows the int64 tables were about 9 GB inside the saved model. The values are unchanged; only the width is.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.qrf import _LeafResidualForest


def test_fitted_forest_keeps_its_lookup_tables_in_int32():
    """The sorted row index and the leaf-offset tables are int32, so they take 4 bytes per entry."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (400, 3))
    y = X[:, 0] + 0.3 * rng.normal(0, 1, 400)
    f = _LeafResidualForest(n_estimators=20, min_samples_leaf=5, random_state=1).fit(X, y)
    assert f._sorted_train_idx_.dtype == np.int32
    assert f._leaf_offsets_.dtype == np.int32
    assert f._sorted_train_idx_.itemsize == 4


def test_fitted_forest_does_not_keep_the_per_row_leaf_matrix():
    """Only the row count and the sorted tables are stored; the (n_train, n_trees) leaf matrix is dropped after fit."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (400, 3))
    y = X[:, 0] + 0.3 * rng.normal(0, 1, 400)
    f = _LeafResidualForest(n_estimators=20, min_samples_leaf=5, random_state=1).fit(X, y)
    assert not hasattr(f, "train_leaves_")
    assert f._n_train_ == 400
