"""Bench: local_linear per-row sklearn Ridge().fit/predict loop vs a single batched numpy normal-equations solve.

local_linear (compute_local_linear_attention) computes, per anchor row, a ridge regression on its top-k ANN
neighbours and emits [intercept, d slopes, r2]. The original hot loop instantiates a fresh sklearn ``Ridge``
object per row and calls ``.fit`` + ``.predict`` -- N_anchor Python-level sklearn dispatches, each solving a
tiny (k x d) problem dominated by object/overhead cost rather than the linear algebra.

The batched path gathers all rows' neighbour blocks into (N, k, d) / (N, k) tensors and solves the centred
normal equations (X_c^T X_c + alpha I) beta = X_c^T y_c for ALL rows in one ``np.linalg.solve`` batch, with the
intercept and r2 computed vectorised. Mathematically identical to sklearn Ridge(fit_intercept=True) up to
float32 reduction order (~1e-7, i.e. float32 ULP; the outputs are cast to float32 anyway).

ANN search (build_hnsw_index/query_topk) is loop-invariant w.r.t. this change: both paths consume the SAME
topk_ids, so the bench feeds identical neighbour blocks to OLD and NEW and times only the regression loop.

Run: CUDA_VISIBLE_DEVICES="" python bench_local_linear_batched_ridge.py
"""
from __future__ import annotations

import time
import numpy as np


def _old_loop(Xn_all, yn_all, d, return_r2, dtype):
    """OLD: per-row sklearn Ridge().fit/predict (verbatim shape of the original inner loop)."""
    from sklearn.linear_model import Ridge
    n_anchor = Xn_all.shape[0]
    n_out_cols = d + 1 + (1 if return_r2 else 0)
    out = np.zeros((n_anchor, n_out_cols), dtype=dtype)
    alpha = 1e-3
    for q in range(n_anchor):
        Xn = Xn_all[q]
        yn = yn_all[q]
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xn, yn)
        out[q, 0] = model.intercept_
        out[q, 1 : 1 + d] = model.coef_
        if return_r2:
            pred = model.predict(Xn)
            ss_res = float(np.sum((yn - pred) ** 2))
            ss_tot = float(np.sum((yn - yn.mean()) ** 2))
            out[q, 1 + d] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return out


def _new_batched(Xn_all, yn_all, d, return_r2, dtype, ridge_alpha=1e-3):
    """NEW: single batched centred normal-equations solve for all rows."""
    n_anchor, k, _ = Xn_all.shape
    n_out_cols = d + 1 + (1 if return_r2 else 0)
    out = np.zeros((n_anchor, n_out_cols), dtype=dtype)
    Xf = Xn_all.astype(np.float64, copy=False)
    yf = yn_all.astype(np.float64, copy=False)
    Xm = Xf.mean(axis=1)  # (N, d)
    ym = yf.mean(axis=1)  # (N,)
    Xc = Xf - Xm[:, None, :]  # (N, k, d)
    yc = yf - ym[:, None]  # (N, k)
    A = np.einsum("nki,nkj->nij", Xc, Xc)  # (N, d, d)
    A[:, np.arange(d), np.arange(d)] += ridge_alpha
    b = np.einsum("nki,nk->ni", Xc, yc)  # (N, d)
    beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (N, d)
    intercept = ym - np.einsum("ni,ni->n", Xm, beta)
    out[:, 0] = intercept.astype(dtype)
    out[:, 1 : 1 + d] = beta.astype(dtype)
    if return_r2:
        pred = np.einsum("nki,ni->nk", Xf, beta) + intercept[:, None]
        ss_res = np.sum((yf - pred) ** 2, axis=1)
        ss_tot = np.sum((yf - ym[:, None]) ** 2, axis=1)
        out[:, 1 + d] = (1.0 - ss_res / np.maximum(ss_tot, 1e-12)).astype(dtype)
    return out


def _make_blocks(n_anchor, k, d, seed=0):
    rng = np.random.default_rng(seed)
    Xn_all = rng.standard_normal((n_anchor, k, d)).astype(np.float32)
    yn_all = rng.standard_normal((n_anchor, k)).astype(np.float32)
    return Xn_all, yn_all


def _best_of(fn, *args, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    dtype = np.float32
    print(f"{'shape (N,k,d)':>20} {'OLD ms':>10} {'NEW ms':>10} {'speedup':>9} {'max|diff|':>12}")
    for n_anchor, k, d in [(2000, 32, 8), (5000, 32, 8), (10000, 32, 8), (5000, 64, 16)]:
        Xn_all, yn_all = _make_blocks(n_anchor, k, d)
        # warm
        _new_batched(Xn_all[:64], yn_all[:64], d, True, dtype)
        old = _old_loop(Xn_all, yn_all, d, True, dtype)
        new = _new_batched(Xn_all, yn_all, d, True, dtype)
        diff = np.abs(old.astype(np.float64) - new.astype(np.float64)).max()
        t_old = _best_of(_old_loop, Xn_all, yn_all, d, True, dtype, repeat=3)
        t_new = _best_of(_new_batched, Xn_all, yn_all, d, True, dtype, repeat=5)
        print(f"{str((n_anchor,k,d)):>20} {t_old*1e3:>10.2f} {t_new*1e3:>10.2f} {t_old/t_new:>8.1f}x {diff:>12.2e}")
