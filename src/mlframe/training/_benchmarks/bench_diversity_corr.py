"""Bench: pairwise diversity correlation between ensemble members.

Compares the DIV-1-COL replacement variants:
- ``pair_loop`` -- O(K^2) Python pair loop, the prior implementation.
- ``numpy_corrcoef`` -- single ``np.corrcoef(M)`` call on the stacked (K, N) matrix.
- ``cupy_corrcoef`` -- ``cupy.corrcoef`` for K>50 OR N>1M (current dispatch threshold).

Usage::

    python -m mlframe.training._benchmarks.bench_diversity_corr
"""

from __future__ import annotations

import time
import numpy as np


def _pair_loop(M: np.ndarray) -> np.ndarray:
    K = M.shape[0]
    out = np.full((K, K), np.nan, dtype=np.float64)
    for i in range(K):
        out[i, i] = 1.0
        for j in range(i + 1, K):
            a, b = M[i], M[j]
            corr = float(np.corrcoef(a, b)[0, 1])
            out[i, j] = corr
            out[j, i] = corr
    return out


def _numpy_full(M: np.ndarray) -> np.ndarray:
    return np.corrcoef(M)


def _bench(shape: tuple, n_repeats: int = 3) -> dict:
    rng = np.random.default_rng(42)
    M = rng.normal(size=shape).astype(np.float64)

    # warm-up
    _pair_loop(M)
    _numpy_full(M)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        a = _pair_loop(M)
    t_pair = (time.perf_counter() - t0) * 1000 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        b = _numpy_full(M)
    t_np = (time.perf_counter() - t0) * 1000 / n_repeats

    assert np.allclose(a, b, atol=1e-9, equal_nan=True)  # nosec B101 - internal invariant check in src/mlframe/training/_benchmarks, not reachable with untrusted input

    # cupy (optional)
    try:
        import cupy as cp

        M_gpu = cp.asarray(M)
        cp.corrcoef(M_gpu)  # warm-up
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            c_gpu = cp.corrcoef(M_gpu)
            c = cp.asnumpy(c_gpu)
        t_cp = (time.perf_counter() - t0) * 1000 / n_repeats
    except Exception:
        t_cp = float("nan")

    return {"shape": shape, "pair_ms": t_pair, "np_ms": t_np, "cp_ms": t_cp}


def main() -> None:
    print("DIV-1-COL bench (averaged over 3 calls; cold cupy excluded)\n")
    print("| shape (K, N) | pair-loop ms | numpy ms | cupy ms | numpy speedup | cupy speedup |")
    print("|---|---|---|---|---|---|")
    for shape in [(5, 1_000), (10, 10_000), (20, 100_000), (50, 100_000), (100, 100_000), (50, 1_000_000)]:
        r = _bench(shape, n_repeats=3)
        np_x = r["pair_ms"] / r["np_ms"] if r["np_ms"] else float("nan")
        cp_x = r["pair_ms"] / r["cp_ms"] if r["cp_ms"] == r["cp_ms"] else float("nan")
        print(f"| {shape} | {r['pair_ms']:.3f} | {r['np_ms']:.3f} | {r['cp_ms']:.3f} | {np_x:.1f}x | {cp_x:.1f}x |")


if __name__ == "__main__":
    main()
