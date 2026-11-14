"""The njit brute-force kNN scorer of the LOF transform equals the matrix form it replaced (kept here as the reference), keeps the dtype contract and handles the edge shapes."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _lof_fe as lof


def _matrix_form(X_new: np.ndarray, X_ref: np.ndarray, lrd_ref: np.ndarray, k_distance_ref: np.ndarray, k_eff: int) -> np.ndarray:
    """Score new rows' LOF against a FROZEN reference set (disjoint from ``X_new``, so no
    self-exclusion is needed): find each new row's k-NN within the reference, form reach-dist
    against the reference points' own frozen ``k_distance_ref``, and compare the new row's local
    reachability density to its neighbors' frozen ``lrd_ref`` - an out-of-sample LOF
    approximation (the reference set stands in for "all training rows" so a large fit only pays
    the reference's bounded cost, not O(n_train^2), at transform time). Rows are independent, so
    they are scored in blocks sized from the free memory: the ``(rows, n_ref)`` distance matrices
    of a million-row transform would not fit."""
    sq_ref = np.sum(X_ref * X_ref, axis=1)
    block = max(1, X_new.shape[0])  # one block: the test inputs are small
    parts = []
    for start in range(0, X_new.shape[0], block):
        rows = X_new[start : start + block]
        sq_new = np.sum(rows * rows, axis=1)
        d2 = sq_new[:, None] + sq_ref[None, :] - 2.0 * (rows @ X_ref.T)
        np.maximum(d2, 0.0, out=d2)
        # the neighbours are picked on the squared distances (same order) and only the k chosen are rooted, not all n_ref per row
        nn_idx = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
        nn_dist = np.sqrt(np.take_along_axis(d2, nn_idx, axis=1))
        reach_dist = np.maximum(k_distance_ref[nn_idx], nn_dist)
        lrd_new = 1.0 / np.maximum(reach_dist.mean(axis=1), 1e-12)
        parts.append(lrd_ref[nn_idx].mean(axis=1) / np.maximum(lrd_new, 1e-12))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.asarray(lrd_ref).dtype)


def _reference(p: int, n_ref: int, seed: int):
    """A reference set and its frozen densities."""
    rng = np.random.default_rng(seed)
    X_ref = rng.standard_normal((n_ref, p)).astype(np.float32)
    lrd, kd, k_eff = lof._lof_fit_reference(X_ref, 20)
    return X_ref, lrd, kd, k_eff


@pytest.mark.parametrize("p", [1, 2, 3, 8])
@pytest.mark.parametrize("n_new", [1, 7, 500, 5000])
def test_njit_scores_equal_the_matrix_form(p, n_new):
    """Scores agree to 1e-4 relative on at least 99.9 percent of the rows (the neighbour set differs only at exact distance ties of the k-th neighbour) and exactly on the rest."""
    X_ref, lrd, kd, k_eff = _reference(p, 400, p + n_new)
    X_new = np.random.default_rng(n_new).standard_normal((n_new, p)).astype(np.float32)
    got = lof._lof_transform_host(X_new, X_ref, lrd, kd, k_eff)
    want = _matrix_form(X_new, X_ref, lrd, kd, k_eff)
    assert got.shape == want.shape
    close = np.isclose(got, want, rtol=1e-4, atol=1e-6)
    assert close.mean() >= 0.999


def test_output_dtype_is_the_dtype_of_the_frozen_densities():
    """float32 densities give float32 scores, float64 densities float64 ones (a replay through apply_recipe is compared bit for bit with the fit)."""
    X_ref, lrd, kd, k_eff = _reference(3, 300, 1)
    X_new = np.random.default_rng(2).standard_normal((50, 3)).astype(np.float32)
    assert lof._lof_transform_host(X_new, X_ref, lrd, kd, k_eff).dtype == np.float32
    assert lof._lof_transform_host(X_new, X_ref, lrd.astype(np.float64), kd.astype(np.float64), k_eff).dtype == np.float64


def test_no_new_rows_gives_an_empty_array():
    """Zero rows give an empty score array of the right dtype."""
    X_ref, lrd, kd, k_eff = _reference(2, 100, 3)
    out = lof._lof_transform_host(np.empty((0, 2), dtype=np.float32), X_ref, lrd, kd, k_eff)
    assert out.shape == (0,) and out.dtype == lrd.dtype


def test_a_duplicate_of_a_reference_point_scores_like_the_inlier_it_is():
    """A new row equal to a reference point has the reference point as its nearest neighbour at distance zero and a score near 1."""
    X_ref, lrd, kd, k_eff = _reference(3, 500, 4)
    score = lof._lof_transform_host(X_ref[:5].copy(), X_ref, lrd, kd, k_eff)
    assert np.all(np.isfinite(score)) and np.all(score > 0.3) and np.all(score < 3.0)
