"""Measurement-only bench for joblib site 5: ``discretization/__init__.py:150``'s
``Parallel(n_jobs=min(8, len(needs_factorize)), prefer="threads")`` pool running
``pd.factorize`` per non-pre-Categorical column, inside ``_multi_col_factorize_native``.

Most categorical columns hit the pre-Categorical ``.cat.codes`` fast path (no joblib
involved at all); this pool only fires for columns that arrive as plain object/string
dtype. Sweeps column count (a handful, matching a realistic "few uncoerced categoricals
out of ~519 initial columns" scenario, up to a worst-case all-object-dtype DataFrame)
and row count (100k matching wellbore scale).

Run: PYTHONPATH=src python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site5_factorize_threading.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.discretization import _multi_col_factorize_native


def _make_df(n, n_cols, seed=0):
    rng = np.random.default_rng(seed)
    cats = [f"cat_{i}" for i in range(50)]
    data = {f"c{j}": rng.choice(cats, size=n) for j in range(n_cols)}
    return pd.DataFrame(data)  # plain object dtype -- forces the factorize path, not .cat.codes


def _best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    n = 99401
    for n_cols, label in ((2, "tiny (2 uncoerced categorical cols -- realistic per-fit count)"),
                          (8, "moderate (8 cols)"),
                          (40, "worst case (40 cols, all object dtype)")):
        df = _make_df(n, n_cols)
        print(f"\n=== {label}, n={n} ===")
        _multi_col_factorize_native(df)  # warm
        t1 = _best_of(lambda: _multi_col_factorize_native(df))
        print(f"n_jobs effectively 1 (len<=1 serial loop only applies at n_cols=1; here comparing the ACTUAL joblib call vs forcing serial):")

        # Serial reference: loop pd.factorize directly (bypassing joblib entirely).
        def _serial():
            out = np.empty((len(df), n_cols), dtype=np.float64)
            for j, c in enumerate(df.columns):
                codes, _ = pd.factorize(df[c], use_na_sentinel=True)
                out[:, j] = codes.astype(np.float64)
            return out

        _serial()
        t_serial = _best_of(_serial)
        t_pool = _best_of(lambda: _multi_col_factorize_native(df))
        print(f"serial python loop: {t_serial*1e3:8.2f} ms   joblib threading pool (n_jobs=min(8,{n_cols})): {t_pool*1e3:8.2f} ms   speedup={t_serial/t_pool:.2f}x")


if __name__ == "__main__":
    main()
