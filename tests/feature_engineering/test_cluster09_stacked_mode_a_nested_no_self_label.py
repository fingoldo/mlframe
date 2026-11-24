"""Cluster 09 regression: a row's own label never changes its own stacked feature in Mode A (audit 2026-11-20, finding 09-F7).

Layer-1 features of training rows used to come from an OOF pass that included the held-out row, so a row's label could shape its own layer-2
neighbourhood. Mode A now rebuilds layer 1 inside each outer fold, so flipping one row's label leaves that row's own layer-2 output unchanged.
"""

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.stacked_qnn import compute_stacked_quantile_neighbours


def _run(X, y):
    """Mode A stacked features with a fixed splitter and seed."""
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    df = compute_stacked_quantile_neighbours(X, y, None, splitter, seed=0, k=5)
    return df.to_numpy()


def test_flipping_a_row_label_leaves_its_own_stacked_feature_unchanged():
    """Row 7's stacked features do not depend on row 7's own label (09-F7)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = rng.normal(size=60)
    base = _run(X, y)
    y2 = y.copy()
    y2[7] = y[7] + 100.0
    flipped = _run(X, y2)
    np.testing.assert_allclose(flipped[7], base[7], rtol=1e-10, atol=1e-12)


def test_output_shape_is_one_row_per_input_row():
    """Control: Mode A still returns one stacked row per input row (09-F7)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = rng.normal(size=60)
    out = _run(X, y)
    assert out.shape[0] == 60
    assert np.all(np.isfinite(out))
