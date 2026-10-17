"""Reference kernels kept beside the fused production paths: weighted aggregates vs numpy, SMOTE row loop vs vectorised."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._aggregation import (
    batch_weighted_mean,
    batch_weighted_std,
    batch_weighted_vector_mean,
    weighted_mean_1d,
    weighted_std_1d,
)
from mlframe.feature_engineering.transformer._smote_kernels_shared import smote_synthesize_rowloop, smote_synthesize_vectorized


def _batch(seed: int, q: int = 7, k: int = 9):
    """Neighbour values and softmax-like weights for a batch of queries."""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal((q, k))
    w = rng.random((q, k))
    return vals, w / w.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_kernels_match_numpy_weighted_moments(seed):
    """Per-query weighted mean and (biased) std equal the numpy weighted formulas; the 1-D kernels agree row by row."""
    vals, w = _batch(seed)
    mean_out = np.empty(vals.shape[0])
    std_out = np.empty(vals.shape[0])
    batch_weighted_mean(vals, w, mean_out)
    batch_weighted_std(vals, w, std_out)
    ref_mean = np.average(vals, weights=w, axis=1)
    ref_std = np.sqrt(np.average((vals - ref_mean[:, None]) ** 2, weights=w, axis=1))
    np.testing.assert_allclose(mean_out, ref_mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(std_out, ref_std, rtol=1e-12, atol=1e-14)
    queries = list(range(vals.shape[0]))
    assert queries
    for q in queries:
        assert weighted_mean_1d(vals[q], w[q]) == pytest.approx(mean_out[q], rel=1e-12)
        assert weighted_std_1d(vals[q], w[q]) == pytest.approx(std_out[q], rel=1e-12)


def test_batch_vector_mean_matches_einsum():
    """The per-query weighted centroid over a (k, dim) block equals the einsum contraction."""
    rng = np.random.default_rng(4)
    vals = rng.standard_normal((6, 8, 5))
    _, w = _batch(4, q=6, k=8)
    out = np.empty((6, 5))
    batch_weighted_vector_mean(vals, w, out)
    np.testing.assert_allclose(out, np.einsum("qk,qkd->qd", w, vals) / w.sum(axis=1, keepdims=True), rtol=1e-12, atol=1e-14)


def test_zero_weights_and_constant_values_are_degenerate_not_nan():
    """Negative controls: all-zero weights give 0.0 (no NaN), identical values give zero spread."""
    vals = np.full((2, 4), 3.0)
    w = np.array([[0.0] * 4, [0.25] * 4])
    mean_out = np.empty(2)
    std_out = np.empty(2)
    batch_weighted_mean(vals, w, mean_out)
    batch_weighted_std(vals, w, std_out)
    assert mean_out[0] == 0.0 and std_out[0] == 0.0
    assert mean_out[1] == pytest.approx(3.0) and std_out[1] == 0.0


@pytest.mark.parametrize("seed", [0, 3])
def test_smote_vectorised_is_bit_identical_to_row_loop(seed):
    """Same RNG draw order, so the vectorised gather+lerp reproduces the row loop exactly."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    a = smote_synthesize_rowloop(X, 55, 5, seed)
    b = smote_synthesize_vectorized(X, 55, 5, seed)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (55, 6) and a.dtype == np.float32
