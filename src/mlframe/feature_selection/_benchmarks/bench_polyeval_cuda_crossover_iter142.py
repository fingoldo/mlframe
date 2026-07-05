"""iter142 microbench: polyeval_dispatch njit / njit_par / cuda crossover on the running host.

Measures the three polyeval backends (``_NJIT_FUNCS`` single-thread Horner, ``_NJIT_PAR_FUNCS`` prange,
``_polyeval_cuda`` RawKernel incl. H2D+D2H) across n in {1e4, 3e4, 1e5, 5e5, 1e6} for the hermite basis,
to find the real njit_par->cuda crossover on THIS GPU and compare it to the source-default ``_CUDA_THRESHOLD``
(500000, derived on an old GTX 1050 Ti and DEFERRED because cupy was broken on the dev box at the time).

The cuda path includes the full host<->device round trip, so it is the honest end-to-end cost the dispatcher
pays. All three backends are bit-identical up to FP reduction-order (Horner is the same recurrence; prange and
the per-thread cuda kernel both evaluate the identical Horner polynomial per element with no cross-element
reduction), so any crossover change is a pure routing recalibration with zero numeric impact.

Run in its own subprocess so a native cupy crash cannot take down the calling session.

Usage:  python -m mlframe.feature_selection._benchmarks.bench_polyeval_cuda_crossover_iter142
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SIZES = (10_000, 30_000, 100_000, 500_000, 1_000_000)
BASIS = "hermite"
REPEATS = 15


def _bench():
    from mlframe.feature_selection.filters.hermite_fe import (
        _NJIT_FUNCS, _NJIT_PAR_FUNCS, _polyeval_cuda, _CUDA_AVAILABLE,
    )
    c = np.array([0.3, -0.7, 0.2, 0.5, -0.1], dtype=np.float64)
    njit = _NJIT_FUNCS[BASIS]
    njit_par = _NJIT_PAR_FUNCS[BASIS]

    # warm numba JIT + cupy NVRTC at a representative size
    xw = np.linspace(-1.0, 1.0, 100_000).astype(np.float64)
    njit(xw, c); njit_par(xw, c)
    if _CUDA_AVAILABLE:
        _polyeval_cuda(BASIS, xw, c)

    def timeit(fn, x):
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn(x, c)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    print(f"basis={BASIS} repeats={REPEATS} cuda_available={_CUDA_AVAILABLE}")
    print(f"{'n':>10} {'njit_ms':>10} {'par_ms':>10} {'cuda_ms':>10} {'winner':>10}")
    rows = []
    for n in SIZES:
        x = np.linspace(-3.0, 3.0, n).astype(np.float64)
        t_njit = timeit(njit, x) * 1e3
        t_par = timeit(njit_par, x) * 1e3
        t_cuda = timeit(lambda xx, cc: _polyeval_cuda(BASIS, xx, cc), x) * 1e3 if _CUDA_AVAILABLE else float("inf")
        cands = {"njit": t_njit, "njit_par": t_par, "cuda": t_cuda}
        winner = min(cands, key=cands.get)
        rows.append((n, t_njit, t_par, t_cuda, winner))
        print(f"{n:>10} {t_njit:>10.3f} {t_par:>10.3f} {t_cuda:>10.3f} {winner:>10}")

    # bit-identity check at the largest size where cuda is plausibly chosen
    if _CUDA_AVAILABLE:
        x = np.linspace(-3.0, 3.0, 1_000_000).astype(np.float64)
        r_par = njit_par(x, c)
        r_cuda = _polyeval_cuda(BASIS, x, c)
        max_abs = float(np.max(np.abs(r_par - r_cuda)))
        print(f"\nbit-identity njit_par vs cuda @1e6: max_abs_diff={max_abs:.3e}")

    # crossover: smallest n where cuda beats njit_par
    cross = None
    for n, _tn, tp, tc, _w in rows:
        if tc < tp:
            cross = n
            break
    print(f"\ncuda-beats-njit_par crossover @this HW: {cross}")
    print(f"source _CUDA_THRESHOLD default = 500000")
    return rows, cross


if __name__ == "__main__":
    _bench()
