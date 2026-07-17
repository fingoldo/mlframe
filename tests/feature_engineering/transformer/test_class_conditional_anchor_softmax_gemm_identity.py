"""Identity-pin for the class_conditional_anchor squared-distance GEMM optimization.

``_softmax_similarity`` and the unified-mass block in ``_compute_class_anchor_features`` were switched
from the ``np.sum((X[:,None,:]-anchors[None,:,:])**2, axis=2)`` broadcast cube to the
``||x||^2 - 2 x.a + ||a||^2`` GEMM form in ``_squared_dists`` (19-46x faster; see
``feature_engineering/_benchmarks/bench_class_conditional_anchor_softmax_gemm.py``).

The change is FP-reduction-order equivalent (selection-safe for the downstream softmax FE features).
This pins:

1. ``_squared_dists`` matches the broadcast reference squared-distance to fp64 truth within float32
   precision -- so a future revert that mis-orders the GEMM operands is caught.
2. The softmax built on ``_squared_dists`` matches the softmax built on the broadcast distances to a
   tight absolute tolerance -- the selection-equivalence guarantee for the emitted features.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.class_conditional_anchor import (
    _softmax_similarity,
    _squared_dists,
)


def _make(n: int, d: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    anchors = rng.standard_normal((k, d)).astype(np.float32)
    return X, anchors


@pytest.mark.parametrize("n,d,k", [(4000, 30, 16), (4000, 30, 32), (8000, 50, 32), (4000, 100, 16)])
def test_squared_dists_close_to_fp64_truth(n, d, k):
    X, anchors = _make(n, d, k)
    got = _squared_dists(X, anchors).astype(np.float64)
    truth = np.sum((X[:, None, :].astype(np.float64) - anchors[None, :, :].astype(np.float64)) ** 2, axis=2)
    rel = np.abs(got - truth) / (np.abs(truth) + 1e-9)
    assert np.max(rel) < 1e-3, f"max rel err vs fp64 {np.max(rel):.3e} (n={n}, d={d}, k={k})"


@pytest.mark.parametrize("n,d,k", [(4000, 30, 16), (4000, 30, 32), (8000, 50, 32), (4000, 100, 16)])
def test_softmax_selection_equivalent_to_broadcast(n, d, k):
    X, anchors = _make(n, d, k, seed=1)
    temp = 1.0
    got = _softmax_similarity(X, anchors, softmax_temp=temp)
    dists_ref = np.sum((X[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
    logits = -dists_ref / (temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    ref = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    assert np.max(np.abs(got - ref)) < 1e-4, f"softmax delta {np.max(np.abs(got - ref)):.3e} (n={n}, d={d}, k={k})"
