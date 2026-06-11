"""cProfile harness for the PR-AUC (average precision) bootstrap CI (charts/binary.py::bootstrap_ap_ci).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_ap_bootstrap_ci``

AP has no closed-form variance (unlike ROC-AUC's DeLong), so the CI is a row bootstrap. The cost is bounded two ways:
the rows are subsampled to ``_AP_BOOTSTRAP_ROW_CAP`` (default 50k) before the draw, and the per-resample AP reuses a
single shared descending sort -- each of B resamples is one ``np.bincount`` scatter + two ``cumsum`` over m rows. At
n=1e6 (subsampled to 50k), B=500 default, this stays sub-second (~0.83s here); the dominant line is the per-B
bincount/cumsum loop, then the shared argsort. Dropping the per-resample concat+diff to a precision-weighted dot
product halved the loop cost.
"""

import cProfile
import pstats
import time

import numpy as np

from mlframe.reporting.charts.binary import bootstrap_ap_ci


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = rng.standard_normal(n) + 1.5 * y
    s = 1.0 / (1.0 + np.exp(-raw))
    return y, s


def main():
    for n in (100_000, 1_000_000):
        y, s = _data(n)
        bootstrap_ap_ci(y, s)  # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            bootstrap_ap_ci(y, s)
        wall = (time.perf_counter() - t0) / 3.0
        print(f"bootstrap_ap_ci @ n={n:,}, B=500 (subsampled 50k): {wall*1000:.1f} ms/call (mean of 3)")

    y, s = _data(1_000_000)
    pr = cProfile.Profile()
    pr.enable()
    bootstrap_ap_ci(y, s)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
