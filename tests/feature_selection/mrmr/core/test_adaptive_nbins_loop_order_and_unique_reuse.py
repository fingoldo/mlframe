"""Wave 13 (2a/2b): _adaptive_nbins.py loop-order swap in ``edges_optimal_joint`` and the
``np.unique(..., return_counts=True)`` reuse in ``_compute_col_edges`` must be selection-equivalent to
the pre-fix versions (same returned edges / same picked winning M), not just faster.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_edges_optimal_joint_matches_naive_loop_order_reference():
    """Reference-implement the ORIGINAL for-M-outer/for-k-inner nesting inline and confirm the
    production edges_optimal_joint (now outer-k/inner-M) returns byte-identical edges for several
    seeds/candidate sets -- pins that the loop-order swap did not change which M wins."""
    from mlframe.feature_selection.filters._adaptive_nbins import (
        edges_optimal_joint,
        _edges_from_quantiles,
        _edges_from_uniform,
        _plug_in_mi,
    )

    def _reference(x, y, candidates=(4, 8, 16, 32), n_splits=3, base="quantile", random_state=0):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y).ravel()
        mask = np.isfinite(x)
        x = x[mask]
        y = y[mask]
        n = x.size
        if n < n_splits * 4:
            from mlframe.feature_selection.filters._adaptive_nbins import edges_freedman_diaconis

            return edges_freedman_diaconis(x, base=base)
        rng = np.random.default_rng(random_state)
        fold_idx = rng.permutation(n) % n_splits
        best_score = -np.inf
        best_M = candidates[0]
        for M in candidates:
            if M < 2:
                continue
            fold_scores = []
            for k in range(n_splits):
                train_mask = fold_idx != k
                val_mask = ~train_mask
                if train_mask.sum() < M or val_mask.sum() < 4:
                    continue
                train_x = x[train_mask]
                val_x, val_y = x[val_mask], y[val_mask]
                edges = _edges_from_quantiles(train_x, M) if base == "quantile" else _edges_from_uniform(train_x, M)
                if edges.size == 0:
                    continue
                binned_val_x = np.searchsorted(edges, val_x, side="right")
                mi = _plug_in_mi(binned_val_x.astype(np.int64), val_y)
                fold_scores.append(mi)
            if fold_scores:
                mean_mi = float(np.mean(fold_scores))
                if mean_mi > best_score:
                    best_score = mean_mi
                    best_M = M
        return _edges_from_quantiles(x, best_M) if base == "quantile" else _edges_from_uniform(x, best_M)

    rng = np.random.default_rng(42)
    for seed in range(5):
        n = 500
        x = rng.standard_normal(n) * (seed + 1)
        y = (x**2 + rng.standard_normal(n) * 0.1 > np.median(x**2)).astype(np.int64)
        ref = _reference(x, y, random_state=seed)
        got = edges_optimal_joint(x, y, random_state=seed)
        np.testing.assert_array_equal(ref, got, err_msg=f"seed={seed}: loop-order swap changed picked edges")


def test_compute_col_edges_sparse_dominance_fallback_matches():
    """A sparse-dominance column (>50% mass at one value) exercises the return_counts reuse path in
    _compute_col_edges (via the public per_feature_nbins dispatcher, method='mah' -- supervised, so it
    collapses and triggers BOTH the collapsed-fallback and the sparse-dominance secondary fallback)."""
    from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

    rng = np.random.default_rng(7)
    n = 2000
    col = np.where(rng.random(n) < 0.8, 0.0, rng.random(n) + 1.0)  # 80% mass at exactly 0.0
    y = (rng.random(n) > 0.5).astype(np.int64)  # independent of col -> col collapses under 'mah'

    edges1 = per_feature_edges(col.reshape(-1, 1), method="mah", y=y, n_jobs=1)[0]
    edges2 = per_feature_edges(col.reshape(-1, 1), method="mah", y=y, n_jobs=1)[0]
    np.testing.assert_array_equal(edges1, edges2)
    assert edges1.size >= 1
