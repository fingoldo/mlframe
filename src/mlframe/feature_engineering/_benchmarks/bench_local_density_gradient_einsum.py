"""Bench: local_density_gradient gradient aggregation — broadcast-temporary vs einsum.

The hot post-kNN block in compute_local_density_gradient_features._process builds
several (n_q, k_eff, d) temporaries:

    diffs     = neighbor_X - Xq_s[:, None, :]      # (n_q, k, d)
    dists     = sqrt((diffs**2).sum(-1)) + eps     # (n_q, k)
    unit_dirs = diffs / dists[:, :, None]          # (n_q, k, d)   <- temp
    gradient  = (weight[:, :, None]   * unit_dirs).mean(1)         # <- temp
    y_gradient= (y_weight[:, :, None] * unit_dirs).mean(1)         # <- temp

Each of the bracketed products materialises a fresh (n_q, k, d) float32 array
(at n_q=100k,k=32,d=50 = 640 MB) only to immediately reduce it over axis 1.
The reduction `(w[:,:,None]*U).mean(1)` is exactly `einsum('qk,qkd->qd', w, U)/k`,
which fuses multiply+reduce with NO (n_q,k,d) temporary.

Identity: einsum sums in the same neighbour order; result is bit-identical for
the gradient/y_gradient (only the two big-temporary products are replaced).

Run: python bench_local_density_gradient_einsum.py
"""
from __future__ import annotations

import time
import numpy as np

EPS = 1e-9


def _old(neighbor_X, Xq_s, neighbor_log_density, log_density_query, neighbor_y):
    diffs = neighbor_X - Xq_s[:, None, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1)) + EPS
    unit_dirs = diffs / dists[:, :, None]
    log_dens_diff = (neighbor_log_density - log_density_query[:, None]).astype(np.float32)
    weight = (log_dens_diff / dists).astype(np.float32)
    gradient = (weight[:, :, None] * unit_dirs).mean(axis=1)
    gradient_norm = np.sqrt((gradient ** 2).sum(axis=-1)).astype(np.float32) + EPS
    y_query_pseudo = neighbor_y.mean(axis=1)
    y_diff = (neighbor_y - y_query_pseudo[:, None]).astype(np.float32)
    y_gradient_weight = (y_diff / dists).astype(np.float32)
    y_gradient = (y_gradient_weight[:, :, None] * unit_dirs).mean(axis=1)
    y_gradient_norm = np.sqrt((y_gradient ** 2).sum(axis=-1)) + EPS
    dot = (gradient * y_gradient).sum(axis=-1)
    alignment = (dot / (gradient_norm * y_gradient_norm)).astype(np.float32)
    return gradient_norm, alignment


def _new(neighbor_X, Xq_s, neighbor_log_density, log_density_query, neighbor_y):
    diffs = neighbor_X - Xq_s[:, None, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1)) + EPS
    unit_dirs = diffs / dists[:, :, None]
    k = unit_dirs.shape[1]
    log_dens_diff = (neighbor_log_density - log_density_query[:, None]).astype(np.float32)
    weight = (log_dens_diff / dists).astype(np.float32)
    gradient = (np.einsum("qk,qkd->qd", weight, unit_dirs, optimize=False) / k).astype(np.float32)
    gradient_norm = np.sqrt((gradient ** 2).sum(axis=-1)).astype(np.float32) + EPS
    y_query_pseudo = neighbor_y.mean(axis=1)
    y_diff = (neighbor_y - y_query_pseudo[:, None]).astype(np.float32)
    y_gradient_weight = (y_diff / dists).astype(np.float32)
    y_gradient = (np.einsum("qk,qkd->qd", y_gradient_weight, unit_dirs, optimize=False) / k).astype(np.float32)
    y_gradient_norm = np.sqrt((y_gradient ** 2).sum(axis=-1)) + EPS
    dot = (gradient * y_gradient).sum(axis=-1)
    alignment = (dot / (gradient_norm * y_gradient_norm)).astype(np.float32)
    return gradient_norm, alignment


def _mk(n_q, k, d, seed=0):
    rng = np.random.default_rng(seed)
    neighbor_X = rng.standard_normal((n_q, k, d)).astype(np.float32)
    Xq_s = rng.standard_normal((n_q, d)).astype(np.float32)
    nld = rng.standard_normal((n_q, k)).astype(np.float32)
    ldq = rng.standard_normal(n_q).astype(np.float32)
    ny = rng.standard_normal((n_q, k)).astype(np.float32)
    return neighbor_X, Xq_s, nld, ldq, ny


def _best_of(fn, args, n=7):
    best = np.inf
    for _ in range(n):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    print(f"{'shape (n_q,k,d)':>22} | {'OLD ms':>9} | {'NEW ms':>9} | {'speedup':>7} | identity")
    for n_q, k, d in [(10_000, 32, 30), (50_000, 32, 50), (100_000, 32, 50)]:
        args = _mk(n_q, k, d)
        o = _old(*args)
        nn = _new(*args)
        gid = np.array_equal(o[0], nn[0])
        aid = np.array_equal(o[1], nn[1])
        # numerical equivalence (einsum vs broadcast may differ in last ULP from reduction order)
        gmax = float(np.nanmax(np.abs(o[0] - nn[0])))
        amax = float(np.nanmax(np.abs(o[1] - nn[1])))
        t_old = _best_of(_old, args) * 1e3
        t_new = _best_of(_new, args) * 1e3
        ident = f"exact={gid and aid} max|d|grad={gmax:.2e} align={amax:.2e}"
        print(f"{str((n_q,k,d)):>22} | {t_old:9.2f} | {t_new:9.2f} | {t_old/t_new:6.2f}x | {ident}")
