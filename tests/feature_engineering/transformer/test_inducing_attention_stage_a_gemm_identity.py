"""Identity-pin for the inducing_attention squared-distance GEMM optimization.

``_stage_a_anchor_to_train`` and ``_stage_b_query_to_anchor`` were switched from the
``((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)`` broadcast cube to the ``||a||^2 - 2 a.b + ||b||^2``
GEMM form in ``_squared_dists`` (12-35x faster; see
``feature_engineering/_benchmarks/bench_inducing_attention_stage_a_gemm.py``). Stage A's cube is (M, N, d)
with N the full train-fold size, so the saving is large.

The change is FP-reduction-order equivalent (selection-safe for the downstream softmax attention features).
This pins the GEMM distance to fp64 truth, the softmax built on it to the broadcast reference, and asserts a
deliberately-wrong GEMM (dropped 2.0 cross-term factor) is caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.inducing_attention import (
    _softmax_with_temp,
    _squared_dists,
    _stage_a_anchor_to_train,
)


def _make(na: int, nb: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((na, d)).astype(np.float32)
    B = rng.standard_normal((nb, d)).astype(np.float32)
    return A, B


@pytest.mark.parametrize("na,nb,d", [(16, 5000, 30), (16, 20000, 30), (32, 8000, 50), (16, 4000, 100)])
def test_squared_dists_close_to_fp64_truth(na, nb, d):
    A, B = _make(na, nb, d)
    got = _squared_dists(A, B).astype(np.float64)
    truth = np.sum((A[:, None, :].astype(np.float64) - B[None, :, :].astype(np.float64)) ** 2, axis=2)
    rel = np.abs(got - truth) / (np.abs(truth) + 1e-9)
    assert np.max(rel) < 1e-3, f"max rel err vs fp64 {np.max(rel):.3e} (na={na}, nb={nb}, d={d})"


@pytest.mark.parametrize("na,nb,d", [(16, 5000, 30), (32, 8000, 50), (16, 4000, 100)])
def test_softmax_selection_equivalent_to_broadcast(na, nb, d):
    A, B = _make(na, nb, d, seed=1)
    temp = 1.0
    got = _softmax_with_temp(-_squared_dists(A, B), temp=temp)
    dists_ref = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    ref = _softmax_with_temp(-dists_ref, temp=temp)
    assert np.max(np.abs(got - ref)) < 1e-4, f"softmax delta {np.max(np.abs(got - ref)):.3e}"


def test_wrong_gemm_dropped_cross_term_factor_is_caught():
    """A GEMM that drops the 2.0 on the cross term must diverge from the broadcast truth -- proves the pin bites."""
    A, B = _make(16, 5000, 30, seed=2)

    def _wrong(A, B):
        a_sq = np.einsum("ij,ij->i", A, A)[:, None]
        b_sq = np.einsum("ij,ij->i", B, B)[None, :]
        d = a_sq - (A @ B.T) + b_sq  # missing the 2.0
        np.maximum(d, 0.0, out=d)
        return d

    truth = np.sum((A[:, None, :].astype(np.float64) - B[None, :, :].astype(np.float64)) ** 2, axis=2)
    rel = np.abs(_wrong(A, B).astype(np.float64) - truth) / (np.abs(truth) + 1e-9)
    assert np.max(rel) > 1e-2


def test_stage_a_runs_and_shapes():
    rng = np.random.default_rng(3)
    anchors = rng.standard_normal((16, 30)).astype(np.float32)
    X = rng.standard_normal((4000, 30)).astype(np.float32)
    y = rng.standard_normal(4000).astype(np.float32)
    y_mean_m, y_std_m = _stage_a_anchor_to_train(anchors, X, y, temp=1.0)
    assert y_mean_m.shape == (16,)
    assert y_std_m.shape == (16,)
    assert np.all(np.isfinite(y_mean_m)) and np.all(y_std_m >= 0)
