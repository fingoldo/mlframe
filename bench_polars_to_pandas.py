"""Two benchmarks on a production-shaped Polars DataFrame (heavy Categoricals):

1. End-to-end CatBoost ``fit`` + ``predict_proba`` with identical
   hyperparameters on (a) the native Polars DataFrame and (b) the same
   data converted to pandas via mlframe's ``get_pandas_view_of_polars_df``.
   This is what actually matters — it shows whether skipping CatBoost's
   internal per-column Polars materialization (``rechunk`` +
   ``to_physical().to_numpy()`` for each Categorical,
   ``_catboost.pyx:3199`` / ``:3288``) buys wall-clock time on a real
   train/predict path. **This is the default mode.**

2. Conversion-only microbench: just the Polars→pandas step (mlframe's
   ``to_arrow`` + batched ``pa.compute.cast`` vs a Python re-implementation
   of CatBoost's per-column loop), without any model call. Useful to
   isolate the conversion cost from model execution.

Run
---
    # End-to-end CatBoost training comparison (default)
    D:/ProgramData/anaconda3/python.exe bench_polars_to_pandas.py

    # Conversion-only microbench
    BENCH_MODE=conversion D:/ProgramData/anaconda3/python.exe bench_polars_to_pandas.py

    # Both
    BENCH_MODE=both D:/ProgramData/anaconda3/python.exe bench_polars_to_pandas.py

Tunables (env vars): ``BENCH_N_ROWS`` (default 200_000), ``BENCH_N_CAT``
(default 70), ``BENCH_N_REPEATS`` (default 3), ``BENCH_ITERATIONS``
(CatBoost iterations, default 50), ``BENCH_THREAD_COUNT`` (CatBoost
thread_count, default -1 = all cores).
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

N_ROWS = int(os.environ.get("BENCH_N_ROWS", "1000000"))
N_CAT = int(os.environ.get("BENCH_N_CAT", "70"))
N_F64 = int(os.environ.get("BENCH_N_F64", "425"))
N_F32 = int(os.environ.get("BENCH_N_F32", "38"))
N_I8 = int(os.environ.get("BENCH_N_I8", "27"))
N_I16 = int(os.environ.get("BENCH_N_I16", "14"))
N_I64 = int(os.environ.get("BENCH_N_I64", "2"))
N_BOOL = int(os.environ.get("BENCH_N_BOOL", "10"))
N_DATETIME = int(os.environ.get("BENCH_N_DATETIME", "1"))
N_REPEATS = int(os.environ.get("BENCH_N_REPEATS", "3"))

MODE = os.environ.get("BENCH_MODE", "catboost").lower()  # catboost | conversion | both
ITERATIONS = int(os.environ.get("BENCH_ITERATIONS", "50"))
THREAD_COUNT = int(os.environ.get("BENCH_THREAD_COUNT", "-1"))
TEST_FRACTION = float(os.environ.get("BENCH_TEST_FRACTION", "0.1"))


def make_synthetic_df(n_rows: int, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols: Dict[str, pl.Series] = {}
    for i in range(N_BOOL):
        cols[f"bool_{i}"] = pl.Series(f"bool_{i}", rng.choice([True, False], n_rows))
    cat_sizes = rng.integers(3, 30, size=N_CAT)
    for i, k in enumerate(cat_sizes):
        pool = [f"c{j}" for j in range(int(k))]
        cols[f"cat_{i}"] = pl.Series(f"cat_{i}", rng.choice(pool, n_rows)).cast(pl.Categorical)
    for i in range(N_DATETIME):
        base_ts = np.datetime64("2024-01-01")
        offsets = rng.integers(0, 365 * 24 * 3600, n_rows).astype("timedelta64[s]")
        cols[f"dt_{i}"] = pl.Series(f"dt_{i}", (base_ts + offsets).astype("datetime64[us]"))
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
    for i in range(N_I64):
        cols[f"i64_{i}"] = pl.Series(f"i64_{i}", rng.integers(0, 1_000_000, n_rows, dtype=np.int64))
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


def _cat_column_names(df: pl.DataFrame) -> List[str]:
    return [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Categorical]


def _make_target(df: pl.DataFrame, seed: int = 0) -> np.ndarray:
    """Build a binary target with some signal from the first two f64 columns."""
    rng = np.random.default_rng(seed)
    f0 = df["f64_0"].to_numpy()
    f1 = df["f64_1"].to_numpy()
    logits = 0.7 * np.nan_to_num(f0) - 0.5 * np.nan_to_num(f1) + rng.standard_normal(len(df)) * 0.3
    return (logits > 0).astype(np.int8)


def run_catboost_end_to_end() -> None:
    """Train CatBoost with identical hyperparameters on (1) native Polars input
    and (2) mlframe's pandas view of the same data. Compare fit and predict_proba
    wall-clock.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("catboost not installed — skipping end-to-end bench.")
        return

    print("=" * 72)
    print("End-to-end CatBoost benchmark (native Polars vs mlframe pandas-view)")
    print(f"  iterations={ITERATIONS}, thread_count={THREAD_COUNT}, "
          f"test_fraction={TEST_FRACTION}, repeats={N_REPEATS}")
    print("=" * 72)

    df_pl = make_synthetic_df(N_ROWS)
    y = _make_target(df_pl)

    # CatBoost doesn't consume Datetime columns directly — drop them, matching
    # the production pipeline (where ``job_posted_at`` is listed in
    # columns_to_drop). This keeps the bench focused on Categorical overhead.
    dt_cols = [c for c, dt in zip(df_pl.columns, df_pl.dtypes) if dt.is_temporal()]
    if dt_cols:
        df_pl = df_pl.drop(dt_cols)

    cat_cols = _cat_column_names(df_pl)

    n_test = max(1, int(N_ROWS * TEST_FRACTION))
    train_pl = df_pl.slice(0, N_ROWS - n_test)
    test_pl = df_pl.slice(N_ROWS - n_test, n_test)
    y_train = y[: N_ROWS - n_test]

    print(f"  train={train_pl.shape}, test={test_pl.shape}, "
          f"cat_features={len(cat_cols)}, dropped_datetime={len(dt_cols)}")

    # Build pandas-view *once per repeat* to fairly include conversion cost.
    timings: Dict[str, Dict[str, List[float]]] = {
        "polars-native":  {"convert": [], "fit": [], "predict": [], "total": []},
        "mlframe-pandas": {"convert": [], "fit": [], "predict": [], "total": []},
    }

    def _build_clf() -> "CatBoostClassifier":
        return CatBoostClassifier(
            iterations=ITERATIONS,
            thread_count=THREAD_COUNT,
            verbose=False,
            allow_writing_files=False,
            random_seed=0,
        )

    for repeat in range(1, N_REPEATS + 1):
        print(f"\n--- Repeat {repeat}/{N_REPEATS} ---")

        # Variant A: native Polars input — CatBoost does its own conversion.
        gc.collect()
        t_total = timer()
        t_conv = 0.0  # no explicit conversion step; CatBoost does it internally
        clf_pl = _build_clf()
        t0 = timer(); clf_pl.fit(train_pl, y_train, cat_features=cat_cols); t_fit = timer() - t0
        t0 = timer(); _ = clf_pl.predict_proba(test_pl); t_pred = timer() - t0
        total_pl = timer() - t_total
        timings["polars-native"]["convert"].append(t_conv)
        timings["polars-native"]["fit"].append(t_fit)
        timings["polars-native"]["predict"].append(t_pred)
        timings["polars-native"]["total"].append(total_pl)
        print(f"  polars-native:  fit={t_fit:6.2f}s  predict={t_pred:6.2f}s  "
              f"total={total_pl:6.2f}s")
        del clf_pl

        # Variant B: mlframe pandas view — explicit conversion then train/predict.
        gc.collect()
        t_total = timer()
        t0 = timer()
        train_pd = get_pandas_view_of_polars_df(train_pl)
        test_pd = get_pandas_view_of_polars_df(test_pl)
        t_conv = timer() - t0
        clf_pd = _build_clf()
        t0 = timer(); clf_pd.fit(train_pd, y_train, cat_features=cat_cols); t_fit = timer() - t0
        t0 = timer(); _ = clf_pd.predict_proba(test_pd); t_pred = timer() - t0
        total_pd = timer() - t_total
        timings["mlframe-pandas"]["convert"].append(t_conv)
        timings["mlframe-pandas"]["fit"].append(t_fit)
        timings["mlframe-pandas"]["predict"].append(t_pred)
        timings["mlframe-pandas"]["total"].append(total_pd)
        print(f"  mlframe-pandas: convert={t_conv:5.2f}s  fit={t_fit:6.2f}s  "
              f"predict={t_pred:6.2f}s  total={total_pd:6.2f}s")
        del clf_pd, train_pd, test_pd

    print("\n" + "=" * 72)
    print(f"SUMMARY (best of {N_REPEATS}):")
    print("=" * 72)
    print(f"{'variant'.ljust(16)}  {'convert':>8}  {'fit':>8}  "
          f"{'predict':>8}  {'total':>8}")
    print("-" * 60)
    rows = []
    for name, t in timings.items():
        row = (name, min(t["convert"]), min(t["fit"]),
               min(t["predict"]), min(t["total"]))
        rows.append(row)
        print(f"{name.ljust(16)}  {row[1]:7.2f}s  {row[2]:7.2f}s  "
              f"{row[3]:7.2f}s  {row[4]:7.2f}s")

    pol_total = rows[0][4]
    pd_total = rows[1][4]
    if pd_total < pol_total:
        print(f"\n  mlframe pandas-view is {pol_total / pd_total:.2f}x faster "
              f"end-to-end on this shape.")
    else:
        print(f"\n  native Polars is {pd_total / pol_total:.2f}x faster "
              f"end-to-end on this shape.")


def run_conversion_benchmarks() -> None:
    run_benchmark()
    run_breakdown()
    run_catboost_breakdown()


if __name__ == "__main__":
    if MODE == "conversion":
        run_conversion_benchmarks()
    elif MODE == "both":
        run_catboost_end_to_end()
        print()
        run_conversion_benchmarks()
    else:  # default: catboost end-to-end
        run_catboost_end_to_end()
