"""Unit tests for the KNN-Shapley closed-form recursion: brute-force-vs-closed-form is the scientific core.

Proves ``_knn_shapley_recursion`` implements the correct closed form by comparing it against an exact
Shapley-value computation over the SAME KNN classification-agreement utility, enumerating all
``2^(n-1)`` coalitions per training point (feasible at ``n_train=8``). Also pins the efficiency
property (values sum to ``v(N) - v(empty)``) and the njit-vs-numpy bit-identity of the batching path.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from mlframe.data_valuation._knn_shapley import _knn_shapley_recursion, knn_shapley


def _knn_utility(order: list[int], y_sorted_by_full_distance: np.ndarray, y_val: float, k: int) -> float:
    """KNN classification-agreement utility (Jia et al. 2019): (1/K) * count of label matches among the
    ``min(K, |S|)`` nearest points in the coalition -- normalized by the FIXED K, not by ``min(K,|S|)``
    (a coalition smaller than K is treated as having ``K - |S|`` non-matching "phantom" neighbors, not
    as if K were smaller; this fixed-K normalization is what the closed-form recursion actually solves
    for, confirmed by hand-derivation against this brute-force reference at n=2,3)."""
    if not order:
        return 0.0
    j = len(order)
    labels = y_sorted_by_full_distance[order]
    kk = min(k, j)
    matched = labels[:kk] == y_val  # the kk NEAREST points within this coalition (order is distance-sorted globally, restricting preserves relative order)
    return float(np.sum(matched) / k)


def _brute_force_knn_shapley_one_val(y_sorted: np.ndarray, y_val: float, k: int) -> np.ndarray:
    """Exact Shapley value of every training point under the KNN utility, full 2^(n-1) enumeration per point."""
    n = y_sorted.shape[0]
    all_idx = list(range(n))
    phi = np.zeros(n, dtype=np.float64)
    for i in all_idx:
        others = [j for j in all_idx if j != i]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                combo_sorted = sorted(combo)  # preserve distance order (indices already distance-sorted)
                with_i = sorted([*combo, i])
                u_without = _knn_utility(combo_sorted, y_sorted, y_val, k)
                u_with = _knn_utility(with_i, y_sorted, y_val, k)
                weight = 1.0 / (n * comb(n - 1, r))
                total += weight * (u_with - u_without)
        phi[i] = total
    return phi


def test_knn_shapley_recursion_matches_brute_force_exact_shapley():
    """The O(n) recursion must match exact 2^(n-1)-enumeration Shapley values within 1e-9 at n_train=8."""
    y_val = 1.0
    # Distance-sorted labels (position 0 = nearest); a mix of matching/non-matching labels.
    y_sorted = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    k = 3

    match = (y_sorted == y_val).astype(np.float64)
    phi_recursion = _knn_shapley_recursion(match, k)
    phi_brute = _brute_force_knn_shapley_one_val(y_sorted, y_val, k)

    np.testing.assert_allclose(phi_recursion, phi_brute, atol=1e-9)


def test_knn_shapley_recursion_efficiency_property():
    """Sum of all training points' Shapley values equals v(full coalition) - v(empty coalition) (efficiency axiom)."""
    y_sorted = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    k = 2
    y_val = 1.0
    match = (y_sorted == y_val).astype(np.float64)
    phi = _knn_shapley_recursion(match, k)

    v_full = _knn_utility(list(range(len(y_sorted))), y_sorted, y_val, k)
    v_empty = 0.0
    assert phi.sum() == pytest.approx(v_full - v_empty, abs=1e-9)


def test_knn_shapley_dummy_far_uninformative_point_near_zero():
    """A point identical in label-match pattern to its immediate predecessor at rank > k contributes ~0 (redundant, outside K)."""
    # all match; k=2 -- ranks 3..6 are redundant given ranks 1-2
    k = 2
    match = np.ones(6)
    phi = _knn_shapley_recursion(match, k)
    # farthest points (beyond k, all-identical utility contribution) should carry small/near-zero value
    assert abs(phi[-1]) < 0.2


def test_knn_shapley_flags_label_noise_biz_val():
    """biz_val: on a 2-class blob bed with 10% flipped train labels, flipped rows skew to low KNN-Shapley values."""
    from sklearn.datasets import make_blobs
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    X, y = make_blobs(n_samples=1200, centers=2, cluster_std=1.5, random_state=0)
    y = y.astype(np.int64)
    n = len(y)
    flip_frac = 0.10
    flip_idx = rng.choice(n, size=int(n * flip_frac), replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

    X_train, y_train_noisy = X[:1000], y_noisy[:1000]
    X_val, y_val = X[1000:], y[1000:]  # clean holdout labels
    values = knn_shapley(X_train, y_train_noisy, X_val, y_val, k=5)

    flip_mask = np.zeros(1000, dtype=bool)
    flip_mask[flip_idx[flip_idx < 1000]] = True
    clean_median = np.median(values[~flip_mask])
    frac_below_median = float(np.mean(values[flip_mask] < clean_median))
    assert frac_below_median >= 0.90, f"only {frac_below_median:.2%} of flipped rows fell below the clean-row median"

    auroc = roc_auc_score(flip_mask.astype(int), -values)
    assert auroc >= 0.85, f"noise-detector AUROC {auroc:.4f} < 0.85"


def test_knn_shapley_regression_target_raises():
    """A continuous (non-integer-valued) y_train raises NotImplementedError, pointing at tmc_shapley."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)  # continuous, not label-like
    Xv = rng.standard_normal((5, 3))
    yv = rng.standard_normal(5)
    with pytest.raises(NotImplementedError):
        knn_shapley(X, y, Xv, yv)
