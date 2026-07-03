"""Backend A/B for ``_pairwise_complete_abs_corr`` (the vectorized pairwise-complete |corr| used by the source-column
dedup) across numpy(BLAS) / njit(parallel) / cupy(CUDA).

Run: ``python -m mlframe.feature_selection.filters._orthogonal_univariate_fe._benchmarks.bench_pairwise_complete_corr``

Context: the dedup pass was O(P^2 * N) per-pair ``np.corrcoef`` on the FULL train frame -- a 1h+ hang on 5M-row well-log
data with many NaN columns. The fix row-caps the correlation rows (``_MAX_CORR_ROWS``, near-constant in N) and
vectorizes the partial-NaN block into this kernel (6 masked matmuls). This bench chooses the kernel backend.

Measured (dev box, RTX-class GPU, cache-warm; kept as the KTC-seed + REJECTED-verdict record):

    n=100k P=150 : numpy  860ms | cupy  316ms (2.72x) | njit_par 2219ms (0.44x)
    n=100k P=300 : numpy 2708ms | cupy 1038ms (2.61x) | njit_par 8414ms (0.36x)
    n=100k P=600 : numpy 6533ms | cupy 11049ms (0.59x, LOSES) | njit_par 31966ms (0.23x)
    n=50k  P=600 : numpy 3557ms | cupy 1723ms (2.06x) | njit_par 16172ms (0.23x)
    n=100k P=40  : numpy  153ms | njit_par 190ms (0.80x)

All backends are bit-identical to numpy (maxdiff ~1e-16). Verdicts:
  * numpy(BLAS)  -- CPU default. Multithreaded MKL/OpenBLAS matmul; hard to beat on CPU.
  * cupy(CUDA)   -- 2.0-2.7x in a moderate band, but LOSES at large P*n (GPU memory pressure, e.g. P=600/n=100k).
                    KTC-gated: chosen only inside its measured winning region, with an OOM auto-fallback to numpy.
  * njit(parallel) -- REJECTED as a default: 0.23-0.80x everywhere (a naive fused triple-loop cannot beat BLAS).
                    Kept as a force-selectable option (MLFRAME_FE_DEDUP_CORR_BACKEND=njit) per REJECTED != DELETED;
                    never auto-selected because it never wins.
"""
from __future__ import annotations

import time

import numpy as np

from .._orth_dedup import _pairwise_complete_abs_corr, _pc_corr_njit


def _cupy_pc(Q, R):
    import cupy as cp

    Q = cp.asarray(Q)
    R = cp.asarray(R)
    Qm = cp.isfinite(Q)
    Rm = cp.isfinite(R)
    Q0 = cp.where(Qm, Q, 0.0)
    R0 = cp.where(Rm, R, 0.0)
    Qmf = Qm.astype(cp.float64)
    Rmf = Rm.astype(cp.float64)
    n = Qmf @ Rmf.T
    Sx = Q0 @ Rmf.T
    Sy = Qmf @ R0.T
    Sxx = (Q0 * Q0) @ Rmf.T
    Syy = Qmf @ (R0 * R0).T
    Sxy = Q0 @ R0.T
    cov = Sxy - Sx * Sy / n
    vx = Sxx - Sx * Sx / n
    vy = Syy - Sy * Sy / n
    corr = cp.abs(cov / cp.sqrt(vx * vy))
    corr[(n < 8) | (vx <= 1e-24) | (vy <= 1e-24)] = cp.nan
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(corr)


def _time(fn, M, rep=3):
    fn(M, M)  # warm
    t = time.perf_counter()
    for _ in range(rep):
        out = fn(M, M)
    return (time.perf_counter() - t) / rep, out


def main():
    try:
        import cupy  # noqa: F401

        has_cupy = True
    except Exception:
        has_cupy = False

    rng = np.random.default_rng(0)
    for n, P in [(100_000, 150), (100_000, 300), (100_000, 600), (50_000, 600), (100_000, 40)]:
        M = rng.normal(size=(P, n))
        M[rng.random((P, n)) < 0.25] = np.nan
        tnp, ref = _time(_pairwise_complete_abs_corr, M)
        tj, cj = _time(_pc_corr_njit, M)
        line = f"n={n:,} P={P}: numpy {tnp * 1e3:7.1f}ms | njit_par {tj * 1e3:7.1f}ms ({tnp / tj:.2f}x)"
        dj = np.nanmax(np.abs(np.nan_to_num(ref) - np.nan_to_num(cj)))
        line += f" [d={dj:.1e}]"
        if has_cupy:
            tg, cg = _time(_cupy_pc, M)
            dg = np.nanmax(np.abs(np.nan_to_num(ref) - np.nan_to_num(cg)))
            line += f" | cupy {tg * 1e3:7.1f}ms ({tnp / tg:.2f}x) [d={dg:.1e}]"
        print(line)


if __name__ == "__main__":
    main()
