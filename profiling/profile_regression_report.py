"""cProfile harness for report_regression_model_perf end-to-end (iter56 loop).

Drives the regression reporter on a modest-n synthetic with print_report/chart
off, so the profile focuses on the metric + extras + residual-audit
orchestration (the regression analog of fast_calibration_report from
iters 54/55). Run:

    PYTHONPATH=src MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" \
        NUMBA_DISABLE_CUDA=1 python profiling/profile_regression_report.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

from mlframe.training.reporting._reporting_regression import report_regression_model_perf


def _make(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    yt = np.abs(rng.standard_normal(n)) * 10.0 + 5.0
    yp = yt + 0.3 * rng.standard_normal(n)
    return yt, yp


def _call(yt, yp):
    m: dict = {}
    report_regression_model_perf(
        targets=yt, columns=["f0", "f1"], model_name="m", model=None,
        preds=yp, metrics=m, print_report=False, show_perf_chart=False,
    )
    return m


def main():
    yt, yp = _make(20_000)
    for _ in range(3):
        _call(yt, yp)  # warm numba

    reps = 60
    t0 = time.perf_counter()
    for _ in range(reps):
        _call(yt, yp)
    wall = (time.perf_counter() - t0) / reps
    print(f"warm e2e: {wall*1e3:.3f} ms/call (n=20000, reps={reps})")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(reps):
        _call(yt, yp)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(35)
    print(s.getvalue())


if __name__ == "__main__":
    main()
