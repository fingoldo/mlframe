"""Benchmark: mlframe's get_pandas_view_of_polars_df VS CatBoost's polars-native
per-column conversion, on a DataFrame shaped like the user's real production set
(mixed dtypes, lots of Categorical columns).

Why this matters
----------------
CatBoost 1.2+ accepts a polars.DataFrame directly in predict_proba/fit. Internally
(_catboost.pyx:3199 and :3288) it iterates columns in Python and, for each
Categorical column, calls ``column.rechunk()`` then ``column.to_physical().to_numpy()``
— a synchronous, single-threaded per-column materialization. For numeric columns
it uses ``__arrow_c_stream__()`` (zero-copy via Arrow).

mlframe's ``get_pandas_view_of_polars_df`` takes a different route:
``df.to_arrow()`` (whole-table at once), one ``pa.compute.cast(col, pa.string())``
per dictionary column (uses Arrow's multi-threaded compute kernels), then
``table.to_pandas()`` once.

This script measures which approach is faster on a realistic shape and reports
per-path breakdown so the bottleneck is visible.

Run
---
    D:/ProgramData/anaconda3/python.exe bench_polars_to_pandas.py

Tune via env vars: ``BENCH_N_ROWS`` (default 200_000), ``BENCH_N_CAT``
(default 70), ``BENCH_N_REPEATS`` (default 3).
"""
from __future__ import annotations

import gc
import os
import statistics
import sys
from time import perf_counter as timer
from typing import Callable, Dict, List, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlframe.training.utils import get_pandas_view_of_polars_df

N_ROWS = int(os.environ.get("BENCH_N_ROWS", "200000"))
N_CAT = int(os.environ.get("BENCH_N_CAT", "70"))
N_F64 = int(os.environ.get("BENCH_N_F64", "425"))
N_F32 = int(os.environ.get("BENCH_N_F32", "38"))
N_I8 = int(os.environ.get("BENCH_N_I8", "27"))
N_I16 = int(os.environ.get("BENCH_N_I16", "14"))
N_BOOL = int(os.environ.get("BENCH_N_BOOL", "10"))
N_REPEATS = int(os.environ.get("BENCH_N_REPEATS", "3"))


def make_synthetic_df(n_rows: int, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols: Dict[str, pl.Series] = {}
    for i in range(N_BOOL):
        cols[f"bool_{i}"] = pl.Series(f"bool_{i}", rng.choice([True, False], n_rows))
    cat_sizes = rng.integers(3, 30, size=N_CAT)
    for i, k in enumerate(cat_sizes):
        pool = [f"c{j}" for j in range(int(k))]
        cols[f"cat_{i}"] = pl.Series(f"cat_{i}", rng.choice(pool, n_rows)).cast(pl.Categorical)
    for i in range(N_F32):
        v = rng.standard_normal(n_rows).astype(np.float32)
        v[rng.random(n_rows) < 0.01] = np.nan
        cols[f"f32_{i}"] = pl.Series(f"f32_{i}", v)
    for i in range(N_F64):
        v = rng.standard_normal(n_rows)
        v[rng.random(n_rows) < 0.01] = np.nan
        cols[f"f64_{i}"] = pl.Series(f"f64_{i}", v)
    for i in range(N_I16):
        cols[f"i16_{i}"] = pl.Series(f"i16_{i}", rng.integers(-500, 500, n_rows, dtype=np.int16))
    for i in range(N_I8):
        cols[f"i8_{i}"] = pl.Series(f"i8_{i}", rng.integers(-50, 50, n_rows, dtype=np.int8))
    return pl.DataFrame(cols)


def mlframe_approach(df: pl.DataFrame):
    """Our method: to_arrow → pa.compute.cast dict→string → to_pandas."""
    return get_pandas_view_of_polars_df(df)


def catboost_like_approach(df: pl.DataFrame) -> Dict[str, np.ndarray | Tuple[np.ndarray, list]]:
    """Mimics _catboost.pyx: per-column Python loop, rechunk per column, and
    to_physical + to_numpy for each Categorical. Returns per-column buffers,
    not a pandas frame, matching the Cython internals.
    """
    out: Dict[str, np.ndarray | Tuple[np.ndarray, list]] = {}
    for col_name in df.columns:
        s = df[col_name]
        rechunked = s.rechunk()
        if rechunked.dtype == pl.Categorical:
            codes = rechunked.to_physical().to_numpy()
            cats = rechunked.cat.get_categories().to_list()
            out[col_name] = (codes, cats)
        else:
            out[col_name] = rechunked.to_numpy()
    return out


def time_block(fn: Callable, *args, **kwargs) -> Tuple[float, object]:
    gc.collect()
    t0 = timer()
    result = fn(*args, **kwargs)
    dt = timer() - t0
    return dt, result


def run_benchmark() -> None:
    print(f"Building synthetic DF: n_rows={N_ROWS:_}, "
          f"bool={N_BOOL}, cat={N_CAT}, f64={N_F64}, f32={N_F32}, "
          f"i16={N_I16}, i8={N_I8}")
    t_build = timer()
    df = make_synthetic_df(N_ROWS)
    print(f"  built in {timer()-t_build:.2f}s, shape={df.shape}, "
          f"estimated size={df.estimated_size() / 1024**2:.1f} MB\n")

    approaches: List[Tuple[str, Callable[[pl.DataFrame], object]]] = [
        ("mlframe (to_arrow + cast + to_pandas)", mlframe_approach),
        ("catboost-like (per-column rechunk + to_physical)", catboost_like_approach),
    ]

    results: Dict[str, List[float]] = {name: [] for name, _ in approaches}
    for repeat in range(1, N_REPEATS + 1):
        print(f"--- Repeat {repeat}/{N_REPEATS} ---")
        for name, fn in approaches:
            dt, _ = time_block(fn, df)
            results[name].append(dt)
            print(f"  {name}: {dt:.3f}s")
        print()

    print("=" * 72)
    print(f"SUMMARY over {N_REPEATS} repeats (best of 3):")
    print("=" * 72)
    rows = []
    for name in results:
        best = min(results[name])
        mean = statistics.mean(results[name])
        rows.append((name, best, mean))
    max_name = max(len(r[0]) for r in rows)
    print(f"{'approach'.ljust(max_name)}   best       mean")
    print("-" * (max_name + 22))
    for name, best, mean in rows:
        print(f"{name.ljust(max_name)}  {best:6.3f}s   {mean:6.3f}s")

    if len(rows) == 2:
        a_name, a_best, _ = rows[0]
        b_name, b_best, _ = rows[1]
        ratio = b_best / a_best if a_best > 0 else float("inf")
        faster, slower = (a_name, b_name) if a_best < b_best else (b_name, a_name)
        print(f"\n  {faster} is {ratio if a_best < b_best else 1/ratio:.2f}x faster "
              f"than {slower} at this scale.")


def run_breakdown() -> None:
    """Measure per-step cost for the mlframe path to locate its own hotspots."""
    print("\n" + "=" * 72)
    print("mlframe breakdown (single run, step-by-step timings):")
    print("=" * 72)
    df = make_synthetic_df(N_ROWS)

    gc.collect()
    t0 = timer(); tbl = df.to_arrow(); t_to_arrow = timer() - t0

    gc.collect()
    t0 = timer()
    fixed_cols = []
    for col in tbl.columns:
        if pa.types.is_dictionary(col.type):
            col = pc.cast(col, pa.string())
        fixed_cols.append(col)
    tbl_fixed = pa.table(fixed_cols, names=tbl.column_names)
    t_cast = timer() - t0

    gc.collect()
    t0 = timer(); pandas_df = tbl_fixed.to_pandas(); t_to_pd = timer() - t0

    total = t_to_arrow + t_cast + t_to_pd
    print(f"  to_arrow():                {t_to_arrow:6.3f}s ({100*t_to_arrow/total:4.1f}%)")
    print(f"  per-column dict→string cast: {t_cast:6.3f}s ({100*t_cast/total:4.1f}%)")
    print(f"  table.to_pandas():          {t_to_pd:6.3f}s ({100*t_to_pd/total:4.1f}%)")
    print(f"  TOTAL:                      {total:6.3f}s")
    print(f"  resulting pandas shape: {pandas_df.shape}")


def run_catboost_breakdown() -> None:
    """Measure per-dtype cost for the catboost-like path."""
    print("\n" + "=" * 72)
    print("catboost-like breakdown (per-column-type, single run):")
    print("=" * 72)
    df = make_synthetic_df(N_ROWS)

    type_totals = {"Categorical": 0.0, "Numeric/Bool": 0.0}
    type_counts = {"Categorical": 0, "Numeric/Bool": 0}

    for col_name in df.columns:
        s = df[col_name]
        gc.collect()
        t0 = timer()
        rechunked = s.rechunk()
        if rechunked.dtype == pl.Categorical:
            _ = rechunked.to_physical().to_numpy()
            _ = rechunked.cat.get_categories().to_list()
            bucket = "Categorical"
        else:
            _ = rechunked.to_numpy()
            bucket = "Numeric/Bool"
        dt = timer() - t0
        type_totals[bucket] += dt
        type_counts[bucket] += 1

    print(f"  Categorical:  {type_totals['Categorical']:6.3f}s over {type_counts['Categorical']:3d} cols  "
          f"(avg {type_totals['Categorical']/max(type_counts['Categorical'],1)*1000:.2f} ms/col)")
    print(f"  Numeric/Bool: {type_totals['Numeric/Bool']:6.3f}s over {type_counts['Numeric/Bool']:3d} cols  "
          f"(avg {type_totals['Numeric/Bool']/max(type_counts['Numeric/Bool'],1)*1000:.2f} ms/col)")
    print(f"  TOTAL:        {sum(type_totals.values()):6.3f}s")


if __name__ == "__main__":
    run_benchmark()
    run_breakdown()
    run_catboost_breakdown()
