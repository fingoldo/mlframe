"""Isolate whether long string contents (not just high cardinality) are
the real cause of the slow polars->pandas conversion in production.

User's `skills_text` / `ontology_skills_text` are likely 100-1000 char
text blobs, not the short synthetic tags used in the earlier bench. Test
whether string length changes the picture.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from mlframe.training.utils import get_pandas_view_of_polars_df


def _build_source(n_rows, n_cat_short, n_cat_long, short_cardinality, long_cardinality,
                  short_len=8, long_len=500, n_numeric=80, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_numeric):
        data[f"f_{i:03d}"] = rng.standard_normal(n_rows).astype(np.float32)

    # short-strings cat columns (typical cat features: country codes, tiers, ...)
    for i in range(n_cat_short):
        pool = np.array([f"s{i}_{j:04d}"[:short_len] for j in range(short_cardinality)])
        data[f"cat_s_{i:02d}"] = pl.Series(
            f"cat_s_{i:02d}",
            pool[rng.integers(0, short_cardinality, size=n_rows)],
        ).cast(pl.Categorical)

    # long-strings cat columns (typical text features: descriptions, blurbs)
    for i in range(n_cat_long):
        # Build pool of long strings — e.g. 500-char blurbs
        base_long = ("lorem ipsum dolor sit amet " * 30)[:long_len]
        pool = np.array([f"{base_long[:-8]}_{j:07d}" for j in range(long_cardinality)])
        data[f"cat_L_{i:02d}"] = pl.Series(
            f"cat_L_{i:02d}",
            pool[rng.integers(0, long_cardinality, size=n_rows)],
        ).cast(pl.Categorical)

    return pl.DataFrame(data)


def _time(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:42s}  {dt:7.2f}s")
    return out, dt


def run_once(label, src, n_rows):
    print(f"\n=== {label}  @ {n_rows:_} rows ===")
    t_b, _ = _time("build source", lambda: src)
    n = src.height
    train, val, test = src.head(int(n * 0.81)), src[int(n*0.81):int(n*0.9)], src[int(n*0.9):]
    _, t_train = _time(f"polars->pandas(train) {train.height:_}", lambda: get_pandas_view_of_polars_df(train))
    _, t_val   = _time(f"polars->pandas(val)   {val.height:_}",   lambda: get_pandas_view_of_polars_df(val))
    _, t_test  = _time(f"polars->pandas(test)  {test.height:_}",  lambda: get_pandas_view_of_polars_df(test))
    total = t_train + t_val + t_test
    print(f"  {'TOTAL':42s}  {total:7.2f}s")
    return total


def main():
    n_rows = 1_000_000

    print("SCENARIO A: short strings only (short_len=8, cardinality 20k)")
    src_a = _build_source(n_rows=n_rows, n_cat_short=4, n_cat_long=0,
                           short_cardinality=20_000, long_cardinality=0,
                           short_len=8)
    total_a = run_once("short-only", src_a, n_rows)
    del src_a

    print("\nSCENARIO B: long strings (long_len=500, cardinality 20k, 4 cols — prod-like)")
    src_b = _build_source(n_rows=n_rows, n_cat_short=0, n_cat_long=4,
                           short_cardinality=0, long_cardinality=20_000,
                           long_len=500)
    total_b = run_once("long-only", src_b, n_rows)
    del src_b

    print("\nSCENARIO C: ULTRA-long strings (long_len=2000, cardinality 50k, 4 cols)")
    src_c = _build_source(n_rows=n_rows, n_cat_short=0, n_cat_long=4,
                           short_cardinality=0, long_cardinality=50_000,
                           long_len=2000)
    total_c = run_once("ultra-long", src_c, n_rows)

    print("\n" + "=" * 60)
    print("SUMMARY (polars->pandas total for 3 splits of 1M-row frame)")
    print("=" * 60)
    print(f"  short strings  (8 ch)   :  {total_a:7.2f}s")
    print(f"  long strings   (500 ch) :  {total_b:7.2f}s  ({total_b/max(total_a,0.01):.2f}x)")
    print(f"  ultra strings  (2000 ch):  {total_c:7.2f}s  ({total_c/max(total_a,0.01):.2f}x)")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main() or 0)
