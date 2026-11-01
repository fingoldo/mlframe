"""The batched KSG scorer draws its noise once for all columns and still equals the per-column call."""

import numpy as np

from mlframe.feature_selection.filters._ksg_cd_fast import ksg_mi_discrete_target, ksg_mi_discrete_target_columns


def test_columns_equal_per_column_calls():
    """Bit-identical to calling the one-column scorer on each column."""
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(np.int64)
    got = ksg_mi_discrete_target_columns(X, y, n_neighbors=5, random_state=3)
    want = np.array([ksg_mi_discrete_target(X[:, [j]], y, n_neighbors=5, random_state=3)[0] for j in range(4)])
    np.testing.assert_array_equal(got, want)
