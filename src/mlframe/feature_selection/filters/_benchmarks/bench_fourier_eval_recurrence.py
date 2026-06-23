"""Bench + identity for _fourier_eval_njit: replace the per-harmonic
math.sin/math.cos calls (2*K transcendentals per sample) with the
angle-addition (Chebyshev) recurrence (2 transcendentals + O(K) mults
per sample).

_fourier_eval_njit is the Fourier-basis eval kernel used by the pair-FE
optimiser (CMA-ES candidate scoring) and the engineered-recipe replay
path. It evaluates ``sum_k a_k*sin(2*pi*k*z) + b_k*cos(2*pi*k*z)``,
k=1..K. The OLD form called math.sin + math.cos K times per sample;
the NEW form computes sin/cos of the base angle 2*pi*z once and steps
the harmonics via
  sin(k*a) = sin((k-1)*a)*cos(a) + cos((k-1)*a)*sin(a)
  cos(k*a) = cos((k-1)*a)*cos(a) - sin((k-1)*a)*sin(a)

Numerical equivalence: ~1 ULP (max-abs-diff ~1e-15 across K=3,5,8),
far below any selection-altering threshold; fastmath is already on for
this kernel so strict IEEE bit-identity was never the contract. Per the
FE selection-equivalence bar (CLAUDE.md), a ~1e-15 reduction-order
delta cannot move which feature is selected.

Measured (this box, py 3.14.3, CUDA off, best-of-20):
  n=2000  K=3 +164%  K=5 +310%  K=8 +534%
  n=50000 K=3 +167%  K=5 +336%  K=8 +530%
i.e. 2.6x-6.3x faster, scaling with K.

Run: python src/mlframe/feature_selection/filters/_benchmarks/bench_fourier_eval_recurrence.py
"""
import math
import time

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _fourier_eval_OLD(z, c):
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    K = c.shape[0] // 2
    if K == 0:
        return out
    two_pi = 2.0 * math.pi
    for i in range(n):
        zi = z[i]
        s = 0.0
        for k in range(1, K + 1):
            ang = two_pi * k * zi
            s += c[2 * (k - 1)] * math.sin(ang)
            s += c[2 * (k - 1) + 1] * math.cos(ang)
        out[i] = s
    return out


def main():
    from mlframe.feature_selection.filters.bases import _fourier_eval_njit as NEW

    rng = np.random.default_rng(1)
    # warm both
    _fourier_eval_OLD(rng.random(8), rng.standard_normal(6))
    NEW(rng.random(8), rng.standard_normal(6))

    for n in (2000, 50000):
        for K in (3, 5, 8):
            z = rng.random(n)
            c = rng.standard_normal(2 * K)
            a = _fourier_eval_OLD(z, c)
            b = NEW(z, c)
            md = float(np.max(np.abs(a - b)))

            def best(f):
                m = 1e9
                for _ in range(20):
                    t = time.perf_counter()
                    f(z, c)
                    m = min(m, time.perf_counter() - t)
                return m * 1e6

            o = best(_fourier_eval_OLD)
            nw = best(NEW)
            print(f"n={n} K={K}: OLD={o:.1f}us NEW={nw:.1f}us {(o / nw - 1) * 100:+.0f}%  maxdiff={md:.2e}")


if __name__ == "__main__":
    main()
