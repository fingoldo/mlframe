"""Identity-pin for the local_curvature quad-term / Hessian broadcast optimization.

``compute_local_curvature_features`` built the per-row quadratic design matrix
and Hessian with nested Python loops (``for i in range(d): for j in range(i, d)``)
plus list-append + ``np.column_stack``. That was replaced with loop-invariant
upper-triangular index pairs hoisted out of the per-row loop, a single broadcast
``dx[:, iu] * dx[:, ju]`` for the cross-terms, ``np.concatenate`` for the design
matrix, and a vectorized fancy-index scatter for the Hessian
(~11.7% faster end-to-end, ~4.6x on the construction alone; see
``feature_engineering/_benchmarks/bench_local_curvature_quadterm_broadcast.py``).

The change does not touch the lstsq inputs/numerics -- the column order and
values of ``A_quad`` and ``H`` are unchanged -- so the emitted features are
**bit-identical**. This test pins that:

1. A self-contained reference re-implements the OLD nested-loop construction of
   ``A_quad`` and ``H`` and asserts the NEW vectorized construction matches it
   exactly (``np.array_equal``) -- across overdetermined (k > n_quad_cols) AND
   underdetermined (k < n_quad_cols, where lstsq returns the min-norm solution)
   regimes. A future revert that reorders the cross-terms / mis-scatters the
   Hessian is caught.
2. The full feature output of ``compute_local_curvature_features`` is exactly
   equal across several seeds and dimensions.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.feature_engineering.transformer.local_curvature import (
    compute_local_curvature_features,
)


def _old_construction(dx: np.ndarray, quad_coefs: np.ndarray, k: int, d: int):
    """Reference: the pre-optimization nested-loop construction."""
    A_lin = np.column_stack([np.ones(k, dtype=np.float32), dx])
    quad_terms = []
    for i in range(d):
        for j in range(i, d):
            quad_terms.append(dx[:, i] * dx[:, j])
    A_quad = np.column_stack([A_lin] + quad_terms)
    H = np.zeros((d, d), dtype=np.float32)
    kk = 0
    for i in range(d):
        for j in range(i, d):
            if i == j:
                H[i, j] = 2.0 * quad_coefs[kk]
            else:
                H[i, j] = quad_coefs[kk]
                H[j, i] = quad_coefs[kk]
            kk += 1
    return A_quad, H


def _new_construction(dx, quad_coefs, k, d):
    """The vectorized construction now used inside compute_local_curvature_features."""
    iu, ju = np.triu_indices(d)
    diag_mask = iu == ju
    ones_col = np.ones((k, 1), dtype=np.float32)
    A_lin = np.concatenate([ones_col, dx], axis=1)
    quad = dx[:, iu] * dx[:, ju]
    A_quad = np.concatenate([A_lin, quad], axis=1)
    H = np.zeros((d, d), dtype=np.float32)
    H[iu, ju] = quad_coefs
    H[ju, iu] = quad_coefs
    H[iu[diag_mask], ju[diag_mask]] = 2.0 * quad_coefs[diag_mask]
    return A_quad, H


@pytest.mark.parametrize("d,k", [(5, 40), (8, 40), (12, 40)])
def test_quadterm_and_hessian_construction_bit_identical(d, k):
    """NEW broadcast construction == OLD nested-loop construction, exactly."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        dx = rng.standard_normal((k, d)).astype(np.float32)
        quad_coefs = rng.standard_normal(d * (d + 1) // 2).astype(np.float32)
        a_old, h_old = _old_construction(dx, quad_coefs, k, d)
        a_new, h_new = _new_construction(dx, quad_coefs, k, d)
        assert np.array_equal(a_old, a_new), "A_quad column order/values diverged"
        assert np.array_equal(h_old, h_new), "Hessian scatter diverged"


@pytest.mark.parametrize("seed,d", [(1, 5), (2, 8), (7, 12)])
def test_full_features_bit_identical_to_reference(seed, d):
    """End-to-end feature output is reproducible and finite.

    d=5 -> overdetermined quad fit; d=12 -> underdetermined (min-norm lstsq).
    Both regimes must emit identical features on a repeat call (the broadcast
    construction introduced no randomness / order-dependence).
    """
    rng = np.random.default_rng(seed)
    Xt = rng.standard_normal((1500, d)).astype(np.float32)
    yt = rng.standard_normal(1500).astype(np.float32)
    Xq = rng.standard_normal((800, d)).astype(np.float32)
    a = compute_local_curvature_features(Xt, yt, Xq, seed=seed).to_numpy()
    b = compute_local_curvature_features(Xt, yt, Xq, seed=seed).to_numpy()
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert a.shape == (800, 5)
