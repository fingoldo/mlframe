"""Boruta reuses one (n, 2p) buffer and draws the same shadows as before (tracker 2026-11-02 wave 16, MEM16-2)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._boruta import boruta_select


def _run(X, rounds=4, seed=3):
    """Run boruta_select with an importance function that records what it was given."""
    seen = []

    def imp(Xs, y):
        """Record a snapshot and the object, return column variances as importances."""
        arr = np.asarray(Xs)
        seen.append((arr.copy(), Xs))
        return arr.var(axis=0)

    boruta_select(X, np.zeros(len(X)), imp, n_iterations=rounds, random_state=seed)
    return seen


def test_shadows_equal_the_per_round_permutation_draws_ndarray():
    """Each round's shadow half equals rng.permuted of the real half with the same seed and draw order."""
    X = np.random.default_rng(0).normal(size=(50, 4))
    seen = _run(X)
    rng = np.random.default_rng(3)
    for snapshot, _ in seen:
        np.testing.assert_array_equal(snapshot[:, :4], X)
        np.testing.assert_array_equal(snapshot[:, 4:], rng.permuted(X, axis=0))


def test_same_buffer_object_is_reused_across_rounds_for_a_single_dtype_frame():
    """The frame given to importance_fn is the same object every round (its buffer refilled in place), and the draws still match."""
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(40, 3)), columns=list("abc"))
    seen = _run(X)
    assert all(frame is seen[0][1] for _, frame in seen)
    rng = np.random.default_rng(3)
    for snapshot, _ in seen:
        np.testing.assert_array_equal(snapshot[:, 3:], rng.permuted(X.to_numpy(), axis=0))
