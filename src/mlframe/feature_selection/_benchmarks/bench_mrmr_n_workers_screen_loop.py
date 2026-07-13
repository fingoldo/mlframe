"""Settles the n_workers=1-vs-parallel dispute from the 2026-07-09 MRMR audit with a real A/B measurement.

Two independent audit passes disagreed on whether ``n_workers=1`` (the MRMR class default) is a footgun
(the whole greedy candidate-confirmation loop in ``screen_predictors`` runs single-threaded regardless of
``n_jobs``) or the already-benchmarked "fast path" (an in-repo comment at
``mrmr/_mrmr_class.py:2820-2823`` claims prior measurement found threading overhead/contention made it
slower). Neither side had a committed benchmark. This one settles it with a real, paired, interleaved A/B
(per the project's A/B methodology: warm, best-of-N, paired/interleaved, selection-identity gate).

Isolates the screen/confirm loop specifically: FE is disabled (``fe_max_steps=0``) so the measurement is
not confounded by feature-engineering wall-time, which dwarfs the screen loop at this row count anyway.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_mrmr_n_workers_screen_loop``
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _make_dataset(n_rows: int, n_cols: int, n_informative: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    coefs = rng.normal(size=n_informative)
    y = X[:, :n_informative] @ coefs + 0.3 * rng.standard_normal(n_rows)
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def _fit_once(X, y, n_workers: int, n_jobs: int, seed: int):
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(
        n_jobs=n_jobs,
        n_workers=n_workers,
        random_seed=seed,
        verbose=0,
        fe_max_steps=0,  # isolate the screen/confirm loop; no FE candidate generation
        dcd_enable=False,  # isolate the core greedy loop from DCD overhead too
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    dt = time.perf_counter() - t0
    support = tuple(sorted(np.where(m.support_)[0].tolist()))
    return dt, support


def main():
    N_ROWS = 300_000
    N_COLS = 150
    N_INFORMATIVE = 8
    N_TRIALS = 5
    N_JOBS = 16

    X, y = _make_dataset(N_ROWS, N_COLS, N_INFORMATIVE, seed=0)
    print(f"Dataset: {N_ROWS} rows x {N_COLS} cols ({N_INFORMATIVE} informative), n_jobs={N_JOBS}")

    # Warm (numba JIT, joblib pool spawn) before timing.
    _fit_once(X, y, n_workers=1, n_jobs=N_JOBS, seed=0)
    _fit_once(X, y, n_workers=8, n_jobs=N_JOBS, seed=0)

    serial_times, parallel_times = [], []
    serial_support, parallel_support = None, None
    print(f"{'trial':>5} {'serial(n_workers=1)':>22} {'parallel(n_workers=8)':>24}")
    for trial in range(N_TRIALS):
        # Interleaved (serial, parallel) order each trial so shared-machine load noise cancels (paired A/B).
        dt_s, sup_s = _fit_once(X, y, n_workers=1, n_jobs=N_JOBS, seed=trial)
        dt_p, sup_p = _fit_once(X, y, n_workers=8, n_jobs=N_JOBS, seed=trial)
        serial_times.append(dt_s)
        parallel_times.append(dt_p)
        if serial_support is None:
            serial_support, parallel_support = sup_s, sup_p
        else:
            assert sup_s == serial_support, "n_workers=1 selection is not seed-stable across trials"
            assert sup_p == parallel_support, "n_workers=8 selection is not seed-stable across trials"
        print(f"{trial:>5} {dt_s:>22.3f} {dt_p:>24.3f}")

    print()
    print(f"serial   median={np.median(serial_times):.3f}s  min={min(serial_times):.3f}s")
    print(f"parallel median={np.median(parallel_times):.3f}s  min={min(parallel_times):.3f}s")
    wins_parallel = sum(1 for s, p in zip(serial_times, parallel_times) if p < s)
    print(f"parallel faster in {wins_parallel}/{N_TRIALS} paired trials")
    speedup = np.median(serial_times) / np.median(parallel_times)
    print(f"median speedup (parallel vs serial): {speedup:.2f}x")
    print(f"selection identical across n_workers: {serial_support == parallel_support} "
          f"(serial selected {len(serial_support)}, parallel selected {len(parallel_support)})")

    print()
    if speedup >= 1.15 and wins_parallel >= (N_TRIALS - 1):
        print("VERDICT: parallel (n_workers>1) meaningfully faster at this shape -- the class default of "
              "n_workers=1 IS a real footgun for wide-enough candidate pools on this hardware.")
    elif speedup <= 0.87 and wins_parallel <= 1:
        print("VERDICT: serial (n_workers=1) meaningfully faster at this shape -- the in-repo 'fast path' "
              "comment is CORRECT for this shape/hardware; the default should stay.")
    else:
        print("VERDICT: no clear winner within noise at this shape -- inconclusive; n_workers default "
              "left UNCHANGED (neither audit claim is strong enough here to justify a default flip).")


if __name__ == "__main__":
    main()
