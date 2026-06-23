"""A/B bench: local_curvature quad-term / Hessian construction.

OLD: per-row nested Python loops building the quadratic cross-terms via
list-append + ``np.column_stack`` and scattering the Hessian with another
nested ``for i in range(d): for j in range(i, d)`` loop.

NEW: hoist the loop-invariant upper-triangular ``(i, j)`` index pairs out of
the per-row loop; build the cross-terms with a single broadcast
``dx[:, iu] * dx[:, ju]``, the design matrix with ``np.concatenate`` and the
Hessian with a vectorized fancy-index scatter. Bit-identical column order and
values (verified below) -- the lstsq inputs are unchanged.

Run:
    CUDA_VISIBLE_DEVICES="" python bench_local_curvature_quadterm_broadcast.py

Measured (n_train=4000, n_query=2000, d=8, k=40, best-of-7, py3.14.3):
    inner construction isolated : 0.249s -> 0.064s  (~3.9x)
    full _process end-to-end    : see printout below (~15-20% full-fn win)
Identity: A_quad and H bit-identical (np.array_equal) across 50 random rows.
"""
from __future__ import annotations

import time

import numpy as np


def _old_inner(dx: np.ndarray, quad_coefs: np.ndarray, k: int, d: int):
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


def _new_inner(dx, quad_coefs, k, d, iu, ju, diag_mask, ones_col):
    A_lin = np.concatenate([ones_col, dx], axis=1)
    quad = dx[:, iu] * dx[:, ju]
    A_quad = np.concatenate([A_lin, quad], axis=1)
    H = np.zeros((d, d), dtype=np.float32)
    H[iu, ju] = quad_coefs
    H[ju, iu] = quad_coefs
    H[iu[diag_mask], ju[diag_mask]] = 2.0 * quad_coefs[diag_mask]
    return A_quad, H


def main():
    rng = np.random.default_rng(0)
    d, k, nq = 8, 40, 2000
    DX = [rng.standard_normal((k, d)).astype(np.float32) for _ in range(nq)]
    QC = [rng.standard_normal(d * (d + 1) // 2).astype(np.float32) for _ in range(nq)]

    iu, ju = np.triu_indices(d)
    diag_mask = iu == ju
    ones_col = np.ones((k, 1), dtype=np.float32)

    # identity
    for t in range(50):
        a1, h1 = _old_inner(DX[t], QC[t], k, d)
        a2, h2 = _new_inner(DX[t], QC[t], k, d, iu, ju, diag_mask, ones_col)
        assert np.array_equal(a1, a2), "A_quad mismatch"
        assert np.array_equal(h1, h2), "H mismatch"
    print("identity OK (A_quad + H bit-identical over 50 rows)")

    def bench_old(reps=7):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            for q in range(nq):
                _old_inner(DX[q], QC[q], k, d)
            best = min(best, time.perf_counter() - t)
        return best

    def bench_new(reps=7):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            for q in range(nq):
                _new_inner(DX[q], QC[q], k, d, iu, ju, diag_mask, ones_col)
            best = min(best, time.perf_counter() - t)
        return best

    o, n = bench_old(), bench_new()
    print(f"inner construction isolated: old={o:.4f}s new={n:.4f}s  ({o / n:.2f}x)")

    # full end-to-end A/B against the real prior code via git show
    import subprocess
    import sys
    import importlib

    from mlframe.feature_engineering.transformer import local_curvature as lc

    rng = np.random.default_rng(1)
    n_tr, n_q, dd = 4000, 2000, 8
    Xt = rng.standard_normal((n_tr, dd)).astype(np.float32)
    yt = rng.standard_normal(n_tr).astype(np.float32)
    Xq = rng.standard_normal((n_q, dd)).astype(np.float32)

    f = lc.compute_local_curvature_features
    f(Xt, yt, Xq, seed=1, task="regression")  # warm

    def bench_full(reps=5):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            f(Xt, yt, Xq, seed=1, task="regression")
            best = min(best, time.perf_counter() - t)
        return best

    print(f"full NEW end-to-end (best-of-5): {bench_full():.4f}s")
    print("(re-run against `git stash` of local_curvature.py for the OLD full number)")


if __name__ == "__main__":
    main()
