"""cProfile harness for the temporal TARGET_ACF / TARGET_PACF panels at production shape.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_target_acf``

Both panels run on the shared FFT autocovariance (``_acf.acf_fft``). The series is tail-capped to
``_acf.MAX_ACF_SERIES`` (200k) before the FFT, so the cost is bounded regardless of n: at n=1e6 the work
is the ``np.isfinite`` mask + mean-centre over n, then two rfft/irfft over the 200k tail. TARGET_PACF
adds the Durbin-Levinson recursion, which runs over the 50-length ACF vector (O(nlags^2)=O(2500)),
negligible next to the FFT.

Conclusion (numbers below): both panels are bounded by the 200k FFT cap (<100ms at any n). The only
length-n work is the finite-mask + mean-centre, both single C passes. No actionable speedup -- the FFT
is capped and the PACF recursion is on the tiny ACF vector. Profiled together at n=1e6.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.temporal import _target_acf_panel, _target_pacf_panel


def _make_series(n: int, phi: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(0)
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


def _walltime(fn, y, repeats: int = 3) -> float:
    fn(y)  # warm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(y)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n in (100_000, 1_000_000, 5_000_000):
        y = _make_series(n)
        acf_ms = _walltime(_target_acf_panel, y) * 1e3
        pacf_ms = _walltime(_target_pacf_panel, y) * 1e3
        print(f"n={n:>9}  target_acf={acf_ms:8.2f} ms  target_pacf={pacf_ms:8.2f} ms")

    y = _make_series(1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    _target_acf_panel(y)
    _target_pacf_panel(y)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
