"""Bench (2026-07): serial-vs-parallel gate for ``_nanminmax_cols`` in compute_naive_outlier_score.

The parallel njit kernel's thread-spawn + (nt, d) reduce buffer + join overhead loses to a plain serial sweep on small frames. Measured crossover
~n*d = 20k: serial 46-83% faster below, parallel wins above. Shipped: ``_nanminmax_cols_serial`` + a ``n*d < 20000`` gate in the dispatcher.

Run: python -m mlframe.preprocessing._benchmarks.bench_nanminmax_serial_gate
"""
import time

import numpy as np

from mlframe.preprocessing.outliers import _nanminmax_cols, _nanminmax_cols_serial


def main():
    X = np.random.rand(1000, 8)
    _nanminmax_cols(X)
    _nanminmax_cols_serial(X)
    for n, d in [(500, 4), (5000, 8), (200000, 30)]:
        Xt = np.random.rand(n, d)
        Xt[Xt < 0.01] = np.nan
        a = _nanminmax_cols(Xt)
        b = _nanminmax_cols_serial(Xt)
        assert np.allclose(a[0], b[0], equal_nan=True) and np.allclose(a[1], b[1], equal_nan=True)  # nosec B101 - internal invariant check in src/mlframe/preprocessing/_benchmarks, not reachable with untrusted input
    print("identity OK")

    def bench(f, X, r=200):
        best = 1e9
        for _ in range(r):
            t = time.perf_counter()
            f(X)
            best = min(best, time.perf_counter() - t)
        return best * 1e6

    for n, d in [(200, 4), (500, 4), (1000, 8), (2000, 8), (5000, 8), (10000, 4), (500, 30), (5000, 30)]:
        X = np.random.rand(n, d)
        p = bench(_nanminmax_cols, X)
        s = bench(_nanminmax_cols_serial, X)
        print(f"n={n} d={d} nd={n*d}: par {p:.1f}us ser {s:.1f}us  ser_faster={100*(1-s/p):.0f}%")


if __name__ == "__main__":
    main()
