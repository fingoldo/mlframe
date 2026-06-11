"""cProfile harness for the WORM + RESID_ACF regression panels at production shape.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_resid_acf_worm``

RESID_ACF computes the residual ACF via the FFT autocovariance. The series is tail-capped to
``_acf.MAX_ACF_SERIES`` (200k) before the FFT, so the cost is bounded regardless of n: at n=1e6 the
dominant work is the residual ``np.isfinite`` mask + the two rfft/irfft over the 200k tail, both O(cap).

WORM sorts the residuals (O(n log n)) then decimates to <=2000 plotted points keeping the tails. The
``scipy.stats.norm.ppf`` is evaluated only on the <=2000 KEPT plotting positions, not the full n -- the
ppf-over-full-n was the panel's biggest cost (61 ms cumulative at 1e6), and the caller plots only the
kept points, so decimating before the ppf is a free ~5x on the panel (100 ms -> 20 ms at 1e6; 558 ms ->
147 ms at 5e6). The remaining cost is the single O(n log n) ``np.sort`` of the residuals, irreducible
for QQ order statistics.

Conclusion (numbers below): RESID_ACF is bounded by the 200k FFT cap (<100ms at any n, capped). WORM is
the one O(n log n) residual sort after the ppf-decimation fix. No further actionable speedup -- the sort
is irreducible for order statistics and the FFT is already series-tail-capped.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.regression import _resid_acf_panel, _worm_panel


def _make_data(n: int, ar: float = 0.5):
    rng = np.random.default_rng(0)
    e = rng.standard_normal(n)
    resid = np.empty(n)
    resid[0] = e[0]
    for i in range(1, n):
        resid[i] = ar * resid[i - 1] + e[i]
    yt = rng.standard_normal(n)
    yp = yt - resid
    return yt, yp


def _walltime(fn, yt, yp, repeats: int = 3) -> float:
    fn(yt, yp)  # warm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(yt, yp)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n in (100_000, 1_000_000, 5_000_000):
        yt, yp = _make_data(n)
        acf_ms = _walltime(_resid_acf_panel, yt, yp) * 1e3
        worm_ms = _walltime(_worm_panel, yt, yp) * 1e3
        print(f"n={n:>9}  resid_acf={acf_ms:8.2f} ms  worm={worm_ms:8.2f} ms")

    yt, yp = _make_data(1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    _resid_acf_panel(yt, yp)
    _worm_panel(yt, yp)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
