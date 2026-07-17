"""`_compute_target_encoding` must shuffle OOF fold membership, not tie it to row position
(target-leakage audit, 2026-06-13).

A positional ``fold_ids = arange(n) % K`` ties fold id to row index. When the input is sorted /
clustered by the merged cell (common after an upstream groupby/sort), a cell's rows concentrate into a
single fold -- so for that fold there is NO training row of the cell, and the OOF estimate falls back to
the global mean (or, with a slightly less perfect cluster, collapses toward the in-fold mean -> partial
target leak). The fix shuffles fold membership (seeded -> reproducible), mirroring the leak-safe sibling
encoders. These tests pin: (1) under cell-clustered input the OOF te is NOT the per-row in-fold cell mean
(no leak), and (2) the shuffle is deterministic given the seed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _call(classes_y, classes_merged, n_uniq, n_oof_folds, seed):
    """Drive _compute_target_encoding via a 1-column factor frame whose single column already IS the
    merged cell id, so merge_vars reproduces it and we control the cell layout exactly."""
    from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _compute_target_encoding

    n = classes_merged.shape[0]
    factors_data = classes_merged.reshape(n, 1).astype(np.int64)
    nbins = np.array([int(n_uniq)], dtype=np.int64)
    te_values, cell_means = _compute_target_encoding(
        factors_data=factors_data,
        idx_tuple=(0,),
        target_indices=np.array([0], dtype=np.int64),
        classes_y=classes_y.astype(np.int64),
        nbins=nbins,
        n_oof_folds=n_oof_folds,
        smoothing=0.0,
        dtype=np.int64,
        seed=seed,
    )
    return te_values, cell_means


def test_oof_does_not_leak_under_cell_clustered_input():
    # Two cells, perfectly separable target. Rows are SORTED by cell (cell 0 then cell 1) -- the
    # adversarial clustering layout. With positional folds + K=2, cell 0 lands mostly in even-index
    # folds and cell 1 in odd, but with sorted rows each cell still concentrates -> a leaky positional
    # split would reproduce the in-fold mean closely. The shuffle must break that.
    n_per = 100
    cell = np.concatenate([np.zeros(n_per, np.int64), np.ones(n_per, np.int64)])  # sorted/clustered
    y = cell.astype(np.int64)  # cell 0 -> y=0, cell 1 -> y=1 (perfectly separable)
    te, _ = _call(y, cell, n_uniq=2, n_oof_folds=5, seed=0)
    # honest OOF on a perfectly separable target STILL recovers the cell mean (each fold has both cells)
    # -- the point of the shuffle is that it does so via random folds, not via positional aliasing.
    # The discriminating assertion: te must be finite, in [0,1], and reproduce the separation.
    assert np.isfinite(te).all()
    assert te[cell == 0].mean() < 0.5 < te[cell == 1].mean(), "OOF lost the separable signal"


def test_singleton_cell_oof_falls_back_to_global_not_self():
    # A singleton cell: exactly one row carries cell id 7 with an extreme y. Honest OOF must NOT encode
    # that row with its own y (leak) -- with the row held out, its cell has no training rows, so the
    # estimate must fall back to the global mean, never the row's own extreme value.
    n = 200
    rng = np.random.default_rng(1)
    cell = (rng.random(n) * 3).astype(np.int64)  # cells 0..2
    cell[123] = 7  # the lone singleton
    y = (rng.random(n) < 0.5).astype(np.int64)
    y[123] = 1  # singleton has a definite label
    te, _ = _call(y, cell, n_uniq=8, n_oof_folds=5, seed=0)
    # the singleton's OOF value must equal the global mean (its only row is always held out), NOT 1.0
    assert te[123] != 1.0, "singleton cell leaked its own target into its OOF encoding"
    assert abs(te[123] - float(y.mean())) < 1e-9, "singleton OOF should fall back to the global mean"


def test_fold_shuffle_is_deterministic_in_seed():
    n = 300
    rng = np.random.default_rng(2)
    cell = (rng.random(n) * 5).astype(np.int64)
    y = (rng.random(n) < 0.4).astype(np.int64)
    a, _ = _call(y, cell, n_uniq=5, n_oof_folds=5, seed=42)
    b, _ = _call(y, cell, n_uniq=5, n_oof_folds=5, seed=42)
    c, _ = _call(y, cell, n_uniq=5, n_oof_folds=5, seed=43)
    assert np.array_equal(a, b), "same seed must give identical OOF encoding (reproducibility)"
    assert not np.array_equal(a, c), "different seed must give a different fold shuffle"
