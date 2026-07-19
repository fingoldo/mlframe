"""Measurement-only bench for joblib site 7: ``stability.py:194``'s
``Parallel(n_jobs=self.n_jobs, backend="threading")`` pool running ``_one_bootstrap``
(subsample + clone + ``estimator.fit``) inside ``StabilityMRMR.fit``.

The source's own comment at ``stability.py:191-193`` already documents the expected
result: "the inner fit holds the GIL anyway so threading doesn't speed up the fits
themselves, but it eliminates the OOM risk [vs loky] and removes loky process-spawn
cost" -- i.e. this pool is NOT expected to win on wall time, only to avoid loky's
memory/spawn overhead. This bench empirically checks that claim using a REAL
``MRMR`` estimator as the wrapped ``self.estimator`` (small config so wall time stays
bench-friendly), at ``n_bootstraps=20`` (class default) and a larger sweep.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site7_stability_bootstrap_threading.py
"""
from __future__ import annotations

import logging
import time

logging.disable(logging.WARNING)

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.stability import StabilityMRMR


def _make_data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    return X, y


def _make_estimator():
    return MRMR(
        quantization_nbins=8, full_npermutations=2, baseline_npermutations=1,
        n_workers=1, verbose=0, use_gpu=False,
    )


def _run(n_jobs, n_bootstraps, X, y):
    sm = StabilityMRMR(estimator=_make_estimator(), n_bootstraps=n_bootstraps, n_jobs=n_jobs, random_state=0)
    t0 = time.perf_counter()
    sm.fit(X, y)
    return time.perf_counter() - t0


def main():
    for n, p, n_bootstraps, label in (
        (500, 10, 12, "small (n_bootstraps=12, n=500/p=10)"),
        (2000, 15, 20, "class default (n_bootstraps=20, n=2000/p=15)"),
    ):
        X, y = _make_data(n, p)
        print(f"\n=== {label} ===", flush=True)
        _run(1, n_bootstraps, X, y)  # warm numba/imports
        t1 = _run(1, n_bootstraps, X, y)
        print(f"n_jobs=1: {t1:6.2f}s", flush=True)
        for n_jobs in (2, 4):
            _run(n_jobs, n_bootstraps, X, y)  # warm pool
            t = _run(n_jobs, n_bootstraps, X, y)
            print(f"n_jobs=1: {t1:6.2f}s   n_jobs={n_jobs}: {t:6.2f}s   speedup={t1/t:.2f}x", flush=True)


if __name__ == "__main__":
    main()
