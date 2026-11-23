"""``ksg_mi_discrete_target`` on a matrix wider than one block goes through column blocks; the ranking of the columns, hence the selected features, is the same as the unblocked call's."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _ksg_cd_fast as K


def test_blocked_mi_keeps_the_column_ranking(monkeypatch):
    """Signal columns stay on top and the values agree to 1e-3 with the single-block (sklearn-stream) result."""
    rng = np.random.default_rng(0)
    n, p = 2000, 40
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, p))
    X[:, 3] += 1.5 * y
    X[:, 17] -= 1.0 * y
    ref = K.ksg_mi_discrete_target(X, y, n_neighbors=3, random_state=0)
    monkeypatch.setattr(K, "_COLUMNS_BLOCK_BYTES", 8 * n * 7)  # 7 columns a block
    blocked = K.ksg_mi_discrete_target(X, y, n_neighbors=3, random_state=0)
    assert blocked.shape == ref.shape
    np.testing.assert_allclose(blocked, ref, atol=1e-3)
    assert set(np.argsort(blocked)[-2:]) == set(np.argsort(ref)[-2:]) == {3, 17}
