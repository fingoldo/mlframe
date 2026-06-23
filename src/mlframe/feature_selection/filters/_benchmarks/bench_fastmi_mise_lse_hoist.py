"""Bench: _mise_optimal_bandwidth LOO log-likelihood h-grid loop hoist.

The MISE-optimal bandwidth grid search in ``_fastmi._mise_optimal_bandwidth``
(the DEFAULT bandwidth for ``fastmi(..., bandwidth="mise")``) recomputed a full
(N,N) ``log_k`` array, a ``log_k.max(axis=1)`` reduction, and a separate (N,N)
``exp`` for each of ``n_grid`` bandwidths.

The row-max of ``log_k[i,j] = -0.5*sp[i,j]/h^2 - C(h)`` is attained at
``argmin_j sp[i,j]`` for EVERY h (coeff is < 0), so ``dmin = sp.min(axis=1)``
and the stabilising shift ``sp - dmin[:,None]`` are loop-invariant. Hoisting
them collapses each iteration to one (N,N) exp + row-sum.

Identity gate: the SELECTED bandwidth (the function's only output) is
bit-identical (the entropy/MI of the caller is a pure function of it). Verified
== across seeds here.

Run:
    CUDA_VISIBLE_DEVICES="" python bench_fastmi_mise_lse_hoist.py

Measured (n=1000, store-python 3.14.3, best-of-20):
    OLD ~379 ms  ->  NEW ~248 ms   (~1.53x, ~35% faster), best_h bit-identical.
"""
from __future__ import annotations

import math
import time

import numpy as np


def _old(zx, zy, n_grid=12, h_min_factor=0.2, h_max_factor=1.5):
    n = zx.size
    h_sil = float(1.0 * (n ** (-1.0 / 6.0)))
    h_grid = np.linspace(h_sil * h_min_factor, h_sil * h_max_factor, n_grid)
    sp = (zx[:, None] - zx[None, :]) ** 2 + (zy[:, None] - zy[None, :]) ** 2
    np.fill_diagonal(sp, np.inf)
    best_h = h_sil
    best_ll = -np.inf
    for h in h_grid:
        log_k = -0.5 * sp / (h * h) - math.log(2.0 * math.pi * h * h)
        m = log_k.max(axis=1)
        f_i = m + np.log(np.exp(log_k - m[:, None]).sum(axis=1))
        f_i = f_i - math.log(n - 1)
        ll = float(np.sum(f_i))
        if ll > best_ll:
            best_ll = ll
            best_h = float(h)
    return best_h


def _new(zx, zy, n_grid=12, h_min_factor=0.2, h_max_factor=1.5):
    n = zx.size
    h_sil = float(1.0 * (n ** (-1.0 / 6.0)))
    h_grid = np.linspace(h_sil * h_min_factor, h_sil * h_max_factor, n_grid)
    sp = (zx[:, None] - zx[None, :]) ** 2 + (zy[:, None] - zy[None, :]) ** 2
    np.fill_diagonal(sp, np.inf)
    dmin = sp.min(axis=1)
    shifted = sp - dmin[:, None]
    log_n_minus_1 = math.log(n - 1)
    best_h = h_sil
    best_ll = -np.inf
    for h in h_grid:
        inv = -0.5 / (h * h)
        c = math.log(2.0 * math.pi * h * h)
        m = dmin * inv - c
        s = np.exp(shifted * inv).sum(axis=1)
        f_i = m + np.log(s) - log_n_minus_1
        ll = float(np.sum(f_i))
        if ll > best_ll:
            best_ll = ll
            best_h = float(h)
    return best_h


def _bench(f, zx, zy, reps=20):
    f(zx, zy)  # warm
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f(zx, zy)
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


def main():
    rng = np.random.default_rng(0)
    n = 1000
    zx = rng.standard_normal(n)
    zy = 0.6 * zx + 0.8 * rng.standard_normal(n)

    ok = True
    for s in range(8):
        r = np.random.default_rng(s)
        a = r.standard_normal(900)
        b = 0.3 * a + r.standard_normal(900)
        if _old(a, b) != _new(a, b):
            ok = False
            print(f"  IDENTITY FAIL seed={s}: {_old(a, b)!r} != {_new(a, b)!r}")
    print(f"best_h bit-identical across 8 seeds: {ok}")

    old_ms = _bench(_old, zx, zy)
    new_ms = _bench(_new, zx, zy)
    print(f"OLD {old_ms:.2f} ms  ->  NEW {new_ms:.2f} ms   ({old_ms / new_ms:.2f}x)")


if __name__ == "__main__":
    main()
