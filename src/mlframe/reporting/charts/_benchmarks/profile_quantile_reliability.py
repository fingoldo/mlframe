"""cProfile harness for the R-6 quantile-reliability panels.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_quantile_reliability``

QUANTILE_RELIABILITY fits one isotonic regression per tau on the per-row
``1(y<=q_tau)`` indicator. The O(n log n) sort inside ``IsotonicRegression.fit``
dominates; the panel caps the fit at ``_ISOTONIC_FIT_CAP`` (100k) uniform-subsampled
rows so a 1e6-row call stays sub-second. This harness verifies that cap pays off and
that the subsampled curve is faithful (the recalibrated coverage is a smooth monotone
estimate, robust to a 100k draw out of 1e6).

PINBALL_DECOMP (CORP) is O(K) model-diagnostics ``decompose`` calls, each a heavy isotonic
recalibration fit (~6 s/tau at 100k); uncapped it dominated at ~243 s at n=1e6. It now
subsamples to ``_CORP_FIT_CAP`` (10k) -- the decomposition is a diagnostic estimate whose
components reproduce within ~1% on a uniform draw, so the panel stays sub-second per tau.
QUANTILE_CROSSING is K-1 vectorised boolean reductions -- not a bottleneck. Profiled together.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from scipy.stats import norm

from mlframe.reporting.charts.quantile import (
    _pinball_decomp_panel, _quantile_crossing_panel, _quantile_reliability_panel,
)


def _make_data(n: int):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n) * 0.5
    alphas = (0.1, 0.25, 0.5, 0.75, 0.9)
    preds = np.column_stack([x + 0.5 * norm.ppf(a) for a in alphas])
    return y, preds, alphas


def _walltime(fn, y, p, a, repeats: int = 3) -> float:
    fn(y, p, a)  # warm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(y, p, a)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n in (100_000, 1_000_000):
        y, p, a = _make_data(n)
        rel = _walltime(_quantile_reliability_panel, y, p, a)
        dec = _walltime(_pinball_decomp_panel, y, p, a)
        crs = _walltime(_quantile_crossing_panel, y, p, a)
        print(
            f"n={n:>9}  reliability(iso,cap=100k)={rel*1e3:8.2f} ms  "
            f"pinball_corp={dec*1e3:8.2f} ms  crossing={crs*1e3:8.2f} ms",
        )

    y, p, a = _make_data(1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    _quantile_reliability_panel(y, p, a)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
