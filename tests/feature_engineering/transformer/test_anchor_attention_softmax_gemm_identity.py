"""Identity-pin for the anchor_attention squared-distance GEMM optimization.

``_score_rows_against_anchors`` (softmax similarity) and the two hard-assignment blocks (train_dists ->
argmin/nanargmin) were switched from the ``np.sum((X[:,None,:]-anchors[None,:,:])**2, axis=2)`` broadcast
cube to the ``||x||^2 - 2 x.a + ||a||^2`` GEMM form in ``_squared_dists`` (19-47x faster; see
``feature_engineering/_benchmarks/bench_anchor_attention_softmax_gemm.py``).

The change is FP-reduction-order equivalent (selection-safe for the downstream softmax FE features and
argmin-identical for the hard-assignment). This pins:

1. ``_squared_dists`` matches the broadcast reference squared-distance to fp64 truth within float32
   precision -- so a future revert that mis-orders the GEMM operands (e.g. drops the 2.0 factor) is caught.
2. The softmax built on ``_squared_dists`` matches the softmax built on the broadcast distances to a
   tight absolute tolerance -- the selection-equivalence guarantee for the emitted similarity features.
3. ``argmin`` over the GEMM distances agrees exactly with ``argmin`` over the broadcast distances -- the
   hard-assignment used for per-anchor aggregates is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.anchor_attention import _squared_dists


def _make(n: int, d: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    anchors = rng.standard_normal((k, d)).astype(np.float32)
    return X, anchors


@pytest.mark.parametrize("n,d,k", [(4000, 30, 32), (8000, 50, 32), (16000, 50, 64), (4000, 100, 32)])
def test_squared_dists_close_to_fp64_truth(n, d, k):
    X, anchors = _make(n, d, k)
    got = _squared_dists(X, anchors).astype(np.float64)
    truth = np.sum((X[:, None, :].astype(np.float64) - anchors[None, :, :].astype(np.float64)) ** 2, axis=2)
    rel = np.abs(got - truth) / (np.abs(truth) + 1e-9)
    assert np.max(rel) < 1e-3, f"max rel err vs fp64 {np.max(rel):.3e} (n={n}, d={d}, k={k})"


@pytest.mark.parametrize("n,d,k", [(4000, 30, 32), (8000, 50, 32), (16000, 50, 64), (4000, 100, 32)])
def test_softmax_selection_equivalent_to_broadcast(n, d, k):
    X, anchors = _make(n, d, k, seed=1)
    temp = 1.0

    def _softmax(dists):
        logits = -dists / (temp + 1e-9)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    got = _softmax(_squared_dists(X, anchors))
    ref = _softmax(np.sum((X[:, None, :] - anchors[None, :, :]) ** 2, axis=2))
    assert np.max(np.abs(got - ref)) < 1e-4, f"softmax delta {np.max(np.abs(got - ref)):.3e} (n={n}, d={d}, k={k})"


@pytest.mark.parametrize("n,d,k", [(4000, 30, 32), (8000, 50, 32), (16000, 50, 64), (4000, 100, 32)])
def test_argmin_hard_assignment_identical_to_broadcast(n, d, k):
    X, anchors = _make(n, d, k, seed=2)
    gemm_assign = np.argmin(_squared_dists(X, anchors), axis=1)
    bcast_assign = np.argmin(np.sum((X[:, None, :] - anchors[None, :, :]) ** 2, axis=2), axis=1)
    assert np.array_equal(gemm_assign, bcast_assign), f"argmin disagreements (n={n}, d={d}, k={k})"
