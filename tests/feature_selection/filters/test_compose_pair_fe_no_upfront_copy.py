"""compose_pair_fe does not copy its input matrix up front (tracker 2026-11-02 wave 16, MEM16-8)."""

import tracemalloc

import numpy as np

from mlframe.feature_selection.filters.composition import compose_pair_fe


def test_zero_rounds_allocate_far_less_than_a_copy_of_the_matrix():
    """With n_rounds=0 nothing is computed; a 16 MB float32 matrix used to be duplicated by X.copy()."""
    X = np.random.default_rng(0).normal(size=(200_000, 20)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    tracemalloc.start()
    out = compose_pair_fe(X, y, n_rounds=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out["X_aug"].shape == X.shape
    assert peak < 0.5 * X.nbytes


def test_input_matrix_is_left_unchanged():
    """The cascade reads the matrix; the caller's array keeps its values."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    before = X.copy()
    compose_pair_fe(X, y, n_rounds=1, top_k_per_round=1, n_trials=3)
    np.testing.assert_array_equal(X, before)
