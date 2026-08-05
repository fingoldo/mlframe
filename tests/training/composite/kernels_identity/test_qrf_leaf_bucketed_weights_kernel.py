"""TRAINING_COMPOSITE_CORE_B-1 (2026-08-05 audit): ``_LeafResidualForest``'s ``_leaf_weights_kernel`` /
``_leaf_weights`` computed QRF membership weights via a FULL O(n_train) linear scan per (query, tree)
pair -- ``for j in range(n_train): if train_leaves[j, t] == leaf`` -- giving O(n_query * n_trees *
n_train) predict cost instead of O(n_query * n_trees * avg_leaf_size). Fixed by building a fit-time
CSR-style leaf-bucketed layout (``_sorted_train_idx_`` / ``_leaf_offsets_``) so predict only visits the
training rows actually in each query's leaf, for both the njit kernel and its numpy fallback.
"""

from __future__ import annotations

import numpy as np
import numba

from mlframe.training.composite.qrf import _LeafResidualForest, _leaf_weights_kernel


def _old_full_scan_kernel(q_leaves, train_leaves, leaf_inv, n_train):
    """Pre-fix reference: the O(n_train)-per-(query,tree) full linear scan the fix replaced."""
    n_query = q_leaves.shape[0]
    n_trees = q_leaves.shape[1]
    w = np.zeros((n_query, n_train), dtype=np.float64)
    for i in numba.prange(n_query):
        for t in range(n_trees):
            leaf = q_leaves[i, t]
            inv = leaf_inv[t, leaf]
            if inv <= 0.0:
                continue
            for j in range(n_train):
                if train_leaves[j, t] == leaf:
                    w[i, j] += inv
    if n_trees > 0:
        w /= n_trees
    return w


_old_full_scan_kernel_njit = numba.njit(parallel=True, cache=False)(_old_full_scan_kernel)


def _fit_forest(n=800, d=4, seed=1, n_estimators=20, min_samples_leaf=5):
    """Fit a small _LeafResidualForest on a synthetic regression problem for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] + rng.standard_normal(n) * 0.3
    m = _LeafResidualForest(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=1)
    m.fit(X, y)
    return m, rng


def test_leaf_bucketed_kernel_bit_identical_to_full_scan_reference():
    """The leaf-bucketed rewrite must produce bit-identical weights to the pre-fix full-scan kernel."""
    m, rng = _fit_forest()
    Xq = rng.standard_normal((50, m.forest_.n_features_in_))
    Xa = np.asarray(Xq, dtype=np.float64)
    q_leaves = np.ascontiguousarray(m.forest_.apply(Xa))
    n_train = m.train_leaves_.shape[0]

    w_new = _leaf_weights_kernel(q_leaves, m._leaf_inv_, m._sorted_train_idx_, m._leaf_offsets_, n_train)
    w_old = _old_full_scan_kernel_njit(q_leaves, m.train_leaves_, m._leaf_inv_, n_train)

    assert np.array_equal(w_new, w_old), "leaf-bucketed weights must be bit-identical to the full-scan reference"


def test_leaf_weights_rows_sum_to_one():
    """Each query row's Meinshausen forest weight must sum to 1 (a valid conditional-distribution
    weighting), sanity-checking the CSR bucketing didn't drop or double-count any training rows."""
    m, rng = _fit_forest(n_estimators=30)
    Xq = rng.standard_normal((40, m.forest_.n_features_in_))
    w = m._leaf_weights(Xq)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_leaf_offsets_partition_every_training_row_exactly_once_per_tree():
    """The CSR bucketing (_sorted_train_idx_ / _leaf_offsets_) must partition all n_train row indices
    exactly once per tree -- no row dropped or duplicated across leaves."""
    m, _ = _fit_forest(n=500, n_estimators=10)
    n_trees = m.train_leaves_.shape[1]
    n_train = m.train_leaves_.shape[0]
    for t in range(n_trees):
        assert m._leaf_offsets_[t, -1] == n_train
        members = m._sorted_train_idx_[t]
        assert sorted(members.tolist()) == list(range(n_train))
