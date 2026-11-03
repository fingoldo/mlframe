"""The minority-row fused kernel must return the original kernel's null maxima bit for bit (binary and multiclass, balanced and imbalanced targets)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _honest_oj_fused as F


def _inputs(rng, n=3000, p=6, n_m=3, n_f=3, n_pool=8, k_y=2, pos=0.5):
    """Random packed-plan and pool inputs of the fused kernels."""
    probs = np.full(k_y, (1.0 - pos) / max(1, k_y - 1))
    probs[0] = pos
    y = rng.choice(k_y, size=n, p=probs / probs.sum()).astype(np.int64)
    fold = rng.permutation(n) % n_f
    nv = np.array([(fold == f).sum() for f in range(n_f)], dtype=np.int64)
    val_idx = np.zeros((n_f, nv.max()), dtype=np.int64)
    for f in range(n_f):
        val_idx[f, : nv[f]] = np.flatnonzero(fold == f)
    kx = rng.integers(2, 9, size=(n_f, p, n_m)).astype(np.int64)
    kx[0, 0, 0] = 0  # an unscored cell
    xb = np.zeros((n_f, p, n_m, nv.max()), dtype=np.int8)
    for f in range(n_f):
        for j in range(p):
            for m in range(n_m):
                xb[f, j, m, : nv[f]] = rng.integers(0, max(1, kx[f, j, m]), size=nv[f])
    rows = np.arange(n, dtype=np.int64)
    pool_dep = np.array([j % p if t < p else -1 for t in range(n_pool)], dtype=np.int64)
    pool_nb = rng.integers(2, 6, size=(n_pool, n_m)).astype(np.int64)
    pool_rc = np.stack([np.stack([rng.integers(0, pool_nb[t, m], size=n) for m in range(n_m)]) for t in range(n_pool)]).astype(np.int32)
    pool_kxe = F._occupied_bins(pool_rc)
    perms = np.stack([rng.permutation(n) for _ in range(16)]).astype(np.int64)
    return perms, y, y.copy(), rows, xb, kx, nv, val_idx, pool_dep, pool_rc, pool_nb, pool_kxe, float(n)


@pytest.mark.parametrize("k_y,pos", [(2, 0.5), (2, 0.05), (4, 0.7), (3, 0.34)])
def test_minority_kernel_equals_original(k_y, pos):
    """Same permutations, same null maxima, bit for bit."""
    a = _inputs(np.random.default_rng(k_y * 100 + int(pos * 100)), k_y=k_y, pos=pos)
    perms, y, yc, rows, xb, kx, nv, val_idx, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff = a
    o1 = np.empty(perms.shape[0])
    o2 = np.empty(perms.shape[0])
    F._fused_block(perms, y, yc, rows, xb, kx, nv, val_idx, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff, o1)
    px = F._px_table(xb, kx, nv)
    F._fused_block_minority(perms, y, yc, rows, xb, kx, nv, val_idx, px, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff, o2)
    assert np.array_equal(o1, o2)
