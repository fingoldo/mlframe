"""Bench: per_column_rff angles/cos/sin broadcast-temporaries vs a fused njit kernel.

OLD path (transformer/per_column_rff.py): builds three full (n, d_input, m) float arrays
``angles`` = X[:,:,None]*W + b, ``cos_part`` = cos(angles), ``sin_part`` = sin(angles), then
a Python ``for j in range(d_input)`` loop copies the per-column cos/sin blocks into the final
interleaved ``out`` (n, d_input*2*m). That is 3 full (n*d_input*m) elementwise passes + an
extra copy pass — pure broadcast-temporary / discarded-work overhead.

NEW path: a single ``@njit(parallel=True)`` kernel ``_pcrff_fused_njit`` that, for each row,
computes ``a = x_j*W[j,i] + b[j,i]`` once and writes ``scale*cos(a)`` / ``scale*sin(a)`` straight
into the interleaved output — no angles/cos_part/sin_part temporaries, no Python copy loop.

Identity: bit-comparison NEW vs OLD on float32. cos/sin are computed on the SAME float32
``a`` value in both paths (OLD: angles is float32 because X_std,W,b are all float32; cos/sin then
astype float32). The fused kernel keeps ``a`` in float32 too, so results are bit-identical up to
libm cos/sin rounding — verified exact == below.

Run: CUDA_VISIBLE_DEVICES="" python bench_per_column_rff_fused_njit.py
"""
from __future__ import annotations

import time

import numba
import numpy as np

from mlframe.feature_engineering.transformer.per_column_rff import compute_per_column_rff


@numba.njit(parallel=True, cache=True, fastmath=False)
def _pcrff_fused_njit(X_std, W, b, scale):  # pragma: no cover - bench
    n, d_input = X_std.shape
    m = W.shape[1]
    out = np.empty((n, d_input * 2 * m), dtype=np.float32)
    for r in numba.prange(n):
        for j in range(d_input):
            base = j * 2 * m
            xj = X_std[r, j]
            for i in range(m):
                a = xj * W[j, i] + b[j, i]
                out[r, base + i] = scale * np.cos(a)
                out[r, base + m + i] = scale * np.sin(a)
    return out


def _old_numpy(X_std, W, b, m):
    """Exact replica of the current per_column_rff numpy body (the OLD side)."""
    n, d_input = X_std.shape
    dtype = np.float32
    angles = X_std[:, :, None] * W[None, :, :] + b[None, :, :]
    scale = float(np.sqrt(1.0 / m))
    cos_part = (scale * np.cos(angles)).astype(dtype, copy=False)
    sin_part = (scale * np.sin(angles)).astype(dtype, copy=False)
    out = np.empty((n, d_input * 2 * m), dtype=dtype)
    for j in range(d_input):
        out[:, j * 2 * m : j * 2 * m + m] = cos_part[:, j, :]
        out[:, j * 2 * m + m : (j + 1) * 2 * m] = sin_part[:, j, :]
    return out


def _bench_one(n, d_input, m, n_iter=20):
    rng = np.random.default_rng(0)
    X_std = rng.standard_normal((n, d_input)).astype(np.float32)
    W = (rng.standard_normal((d_input, m))).astype(np.float32)
    b = (rng.uniform(0, 2.0 * np.pi, size=(d_input, m))).astype(np.float32)
    scale = np.float32(np.sqrt(1.0 / m))

    old = _old_numpy(X_std, W, b, m)
    new = _pcrff_fused_njit(X_std, W, b, scale)  # warm JIT
    exact = np.array_equal(old, new)
    max_abs = float(np.abs(old.astype(np.float64) - new.astype(np.float64)).max())

    def timeit(fn):
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_old = timeit(lambda: _old_numpy(X_std, W, b, m))
    t_new = timeit(lambda: _pcrff_fused_njit(X_std, W, b, scale))
    print(f"n={n:>7} d={d_input:>3} m={m}: OLD={t_old*1e3:8.3f}ms NEW={t_new*1e3:8.3f}ms " f"speedup={t_old/t_new:5.2f}x  exact=={exact} max_abs={max_abs:.2e}")
    return exact


if __name__ == "__main__":
    print("== per_column_rff fused-njit A/B (best-of-N min wall) ==")
    all_exact = True
    for n in (2000, 20000, 100000, 500000):
        for d_input, m in ((8, 4), (20, 4), (50, 8)):
            all_exact &= _bench_one(n, d_input, m)
    print("ALL EXACT-IDENTICAL:", all_exact)
