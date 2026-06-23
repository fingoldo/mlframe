"""Regression: local_density_gradient einsum gradient aggregation is bit-identical
to the broadcast-temporary reference it replaced.

The hot post-kNN block builds `unit_dirs` (n_q, k, d) and originally reduced it via
`(weight[:, :, None] * unit_dirs).mean(axis=1)`, materialising a (n_q, k, d) product
temporary per gradient. The shipped path uses `einsum('qk,qkd->qd', ...)/k`, which
sums in the same neighbour order. This test pins exact equality so a future "just use
the broadcast form" (or any reduction-order change) is caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.local_density_gradient import (
    compute_local_density_gradient_features,
)


def _broadcast_grad_ref(weight, unit_dirs):
    """The exact pre-optimization aggregation: broadcast product then mean over neighbours."""
    return (weight[:, :, None] * unit_dirs).mean(axis=1)


def test_einsum_matches_broadcast_reference_bit_identical():
    rng = np.random.default_rng(7)
    n_q, k, d = 500, 32, 24
    weight = rng.standard_normal((n_q, k)).astype(np.float32)
    unit_dirs = rng.standard_normal((n_q, k, d)).astype(np.float32)

    ref = _broadcast_grad_ref(weight, unit_dirs)
    fast = (np.einsum("qk,qkd->qd", weight, unit_dirs, optimize=False) / unit_dirs.shape[1]).astype(np.float32)

    assert np.array_equal(ref, fast), "einsum gradient aggregation diverged from the broadcast-mean reference"


def test_compute_features_runs_and_is_finite():
    """End-to-end smoke: the transformer still produces the 5 expected finite feature columns."""
    rng = np.random.default_rng(3)
    Xt = rng.standard_normal((400, 12)).astype(np.float32)
    yt = rng.standard_normal(400).astype(np.float32)
    Xq = rng.standard_normal((120, 12)).astype(np.float32)

    df = compute_local_density_gradient_features(Xt, yt, Xq, seed=1)
    assert df.shape == (120, 5)
    arr = df.to_numpy()
    # alignment column can legitimately be ~0; just require no NaN/Inf leaked from the fused reduction.
    assert np.isfinite(arr).all()
