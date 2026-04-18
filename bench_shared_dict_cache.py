"""Benchmark: profile get_pandas_view_of_polars_df on production-sized data.

Earlier attempt at a shared-dict cache optimization (2026-04-19) was
reverted because Polars trims each slice's Categorical dictionary to the
values actually present in that slice — so train/val/test dicts differ
and cross-call sharing never kicks in.

This bench replaces that cache experiment with a profiling run that
splits the conversion into its sub-steps (to_arrow / dict-rebuild /
to_pandas) so we can tell where the production 100+ s per split actually
goes, and whether a real optimization is possible.

Run:
    python bench_shared_dict_cache.py --n-rows 1000000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import polars as pl


def _build_source(
    n_rows: int,
    n_low_card: int = 9,
    n_high_card: int = 4,
    low_cardinality: int = 50,
    high_cardinality: int = 20_000,
    n_numeric: int = 80,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a Polars frame approximating the production shape:
    13 Categorical columns (4 high-card), 80 numeric columns, 1M rows."""
    rng = np.random.default_rng(seed)
    data: dict = {}

    for i in range(n_numeric):
        data[f"f_{i:03d}"] = rng.standard_normal(n_rows).astype(np.float32)

    for i in range(n_low_card):
        pool = np.array([f"c{i}_{j}" for j in range(low_cardinality)])
        data[f"cat_lo_{i:02d}"] = pl.Series(
            f"cat_lo_{i:02d}",
            pool[rng.integers(0, low_cardinality, size=n_rows)],
        ).cast(pl.Categorical)

    for i in range(n_high_card):
        pool = np.array([f"t{i}_{j:05d}" for j in range(high_cardinality)])
        data[f"cat_hi_{i:02d}"] = pl.Series(
            f"cat_hi_{i:02d}",
            pool[rng.integers(0, high_cardinality, size=n_rows)],
        ).cast(pl.Categorical)

    return pl.DataFrame(data)


def _split_81_9_10(src):
    n = src.height
    i1 = int(n * 0.81)
    i2 = int(n * 0.90)
    return src.head(i1), src[i1:i2], src[i2:]


def _time_step(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"    {label:50s}  {dt:7.2f}s")
    return out, dt


# ---------------------------------------------------------------------------
# The current (as-shipped) path, with per-step timing.
# ---------------------------------------------------------------------------

def convert_current_instrumented(df: pl.DataFrame, label: str):
    print(f"  [{label}] rows={df.height:_}  cols={df.width}")
    tbl, t_arrow = _time_step("1. df.to_arrow()", lambda: df.to_arrow())

    def _rebuild_dicts():
        fixed = []
        for col in tbl.columns:
            if pa.types.is_dictionary(col.type):
                chunks = []
                for chunk in col.chunks:
                    idx32 = pc.cast(chunk.indices, pa.int32())
                    chunks.append(pa.DictionaryArray.from_arrays(idx32, chunk.dictionary))
                col = pa.chunked_array(chunks)
            fixed.append(col)
        return pa.table(fixed, names=tbl.column_names)
    tbl_fixed, t_rebuild = _time_step("2. dict-rebuild (u32->i32 indices)", _rebuild_dicts)

    df_pd, t_to_pandas = _time_step("3. tbl.to_pandas()", lambda: tbl_fixed.to_pandas())

    total = t_arrow + t_rebuild + t_to_pandas
    print(f"    {'[TOTAL]':50s}  {total:7.2f}s")
    return df_pd, total


# ---------------------------------------------------------------------------
# Alternative A: direct-Polars path — skip pyarrow for Categorical columns,
# build pd.Categorical from_codes using polars physical codes + categories.
# ---------------------------------------------------------------------------

def convert_direct_polars(df: pl.DataFrame, label: str):
    """Build a pandas DataFrame column-by-column, using polars' native
    accessors for Categorical columns to skip the pyarrow dict round-trip.
    Non-Categorical columns still go through arrow (zero-copy).
    """
    print(f"  [{label}] rows={df.height:_}  cols={df.width}")
    t0_total = time.perf_counter()

    # Split columns by type.
    cat_cols = [c for c in df.columns if df[c].dtype == pl.Categorical]
    non_cat_cols = [c for c in df.columns if c not in cat_cols]

    def _non_cat():
        sub = df.select(non_cat_cols)
        tbl = sub.to_arrow()
        return tbl.to_pandas()
    non_cat_pd, t_non_cat = _time_step("1. non-cat cols via arrow", _non_cat)

    def _cat_cols_build():
        blocks = {}
        for c in cat_cols:
            s = df[c]
            codes = s.to_physical().to_numpy().astype(np.int32, copy=False)
            categories = s.cat.get_categories().to_list()
            blocks[c] = pd.Categorical.from_codes(codes, categories=categories)
        return pd.DataFrame(blocks, index=non_cat_pd.index)
    cat_pd, t_cat_build = _time_step("2. cat cols direct-from-codes", _cat_cols_build)

    def _concat():
        # Preserve original column order
        out = pd.concat([non_cat_pd, cat_pd], axis=1)
        return out[list(df.columns)]
    df_pd, t_concat = _time_step("3. concat + reorder", _concat)

    total = time.perf_counter() - t0_total
    print(f"    {'[TOTAL]':50s}  {total:7.2f}s")
    return df_pd, total


# ---------------------------------------------------------------------------
# Alternative B: cache pandas CategoricalDtype across splits (not the pyarrow
# dict but the pandas-level CategoricalDtype). When a later split shares the
# same categories list (even if polars dict trimmed them and the arrow dict
# differs), we can avoid rebuilding the pandas CategoricalDtype by using a
# shared one. This is useful if CategoricalDtype construction is the slow
# bit — which we'll discover from the direct-polars bench.
# ---------------------------------------------------------------------------


def run(n_rows: int):
    print("=" * 80)
    print(f"get_pandas_view_of_polars_df profile @ {n_rows:_} rows")
    print("=" * 80)

    t0 = time.perf_counter()
    src = _build_source(n_rows=n_rows)
    print(f"[setup] source {src.shape}, build: {time.perf_counter() - t0:.1f}s")
    train, val, test = _split_81_9_10(src)
    print(f"[setup] split: train={train.height:_}  val={val.height:_}  test={test.height:_}")

    # Warmup so the first call doesn't carry pyarrow init / JIT overhead.
    # Can't use train.to_arrow().to_pandas() directly because pyarrow refuses
    # polars' uint32 dict indices (the exact problem get_pandas_view_of_polars_df
    # works around). Warm up via a small numeric-only slice.
    print("\n[warmup]")
    _ = train.head(100).select([c for c in train.columns if not c.startswith("cat_")]).to_arrow().to_pandas()

    # Current path, per-step.
    print("\n[current path] instrumented per-step")
    _, t_train_cur = convert_current_instrumented(train, "train")
    _, t_val_cur   = convert_current_instrumented(val, "val")
    _, t_test_cur  = convert_current_instrumented(test, "test")
    print(f"\n  CURRENT PATH TOTAL: {t_train_cur + t_val_cur + t_test_cur:.2f}s")

    # Alternative: direct-polars path.
    print("\n[direct-polars path] avoid pyarrow for cat columns")
    _, t_train_dir = convert_direct_polars(train, "train")
    _, t_val_dir   = convert_direct_polars(val, "val")
    _, t_test_dir  = convert_direct_polars(test, "test")
    dir_total = t_train_dir + t_val_dir + t_test_dir
    cur_total = t_train_cur + t_val_cur + t_test_cur
    print(f"\n  DIRECT PATH TOTAL:  {dir_total:.2f}s  ({cur_total / dir_total:.2f}x vs current)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  current (to_arrow+rebuild+to_pandas): train={t_train_cur:.2f}s  val={t_val_cur:.2f}s  test={t_test_cur:.2f}s  TOTAL={cur_total:.2f}s")
    print(f"  direct  (polars codes + arrow rest):  train={t_train_dir:.2f}s  val={t_val_dir:.2f}s  test={t_test_dir:.2f}s  TOTAL={dir_total:.2f}s")
    if dir_total < cur_total:
        saved = cur_total - dir_total
        print(f"  SAVED: {saved:.2f}s  ({cur_total / dir_total:.2f}x)")
    else:
        print(f"  direct is {dir_total / cur_total:.2f}x SLOWER; not a win")
    print("=" * 80)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rows", type=int, default=500_000)
    args = p.parse_args()
    run(args.n_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
