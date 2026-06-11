"""cProfile harness for DeLong AUC CI (charts/calibration.py::delong_auc_ci).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_delong_auc_ci``

DeLong's AUC variance is closed-form O(n log n): the whole cost is three
``np.argsort`` midrank passes (pooled, positives, negatives) plus O(n) vector
arithmetic for the structural components V10 / V01. No bootstrap, no resampling.
Optimization trail (n=1e6): 1769 ms -> 415 ms (-77%, 4.3x). The original midrank
used a python tie-block while-loop over the n sorted values, which was ~93% of
wall at n=1e6 (7.0s of 7.5s in cProfile) -- a per-element python loop, the exact
anti-pattern the efficiency mandate forbids. Replaced by a vectorised two-pass
searchsorted (left/right tie bounds) so the midrank is pure C O(n log n). AUC
stays bit-identical to sklearn roc_auc_score (incl. ties). The residual wall is
the searchsorted + argsort passes (the O(n log n) DeLong floor) -- a bootstrap CI
would multiply this by B (~1000x), which is why DeLong is the right tool for AUC.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.calibration import delong_auc_ci, delong_auc_variance


def _make_data(n: int, sep: float = 1.0):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    s = rng.standard_normal(n) + sep * y
    return y, s


def main(n: int = 1_000_000):
    y, s = _make_data(n)
    delong_auc_variance(y, s)  # warmup

    t0 = time.perf_counter()
    for _ in range(5):
        delong_auc_ci(y, s)
    wall = (time.perf_counter() - t0) / 5.0
    print(f"delong_auc_ci @ n={n}: {wall*1000:.1f} ms/call (best-of-5 mean)")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        delong_auc_ci(y, s)
    pr.disable()
    s_out = StringIO()
    pstats.Stats(pr, stream=s_out).sort_stats("tottime").print_stats(12)
    print(s_out.getvalue())


if __name__ == "__main__":
    main()
