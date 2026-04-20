r"""Trace ``category``'s physical_codes_max through each step of the
production pipeline to find exactly when the codes go sparse.

Baseline run of ``reproduce_xgb_inmemory.py`` on prod box reported
``category physical_codes_range=[2, 3287945]`` right before XGB.fit().
The minimizer's first cast step (``pl.col(Utf8).cast(pl.Categorical)``)
gives max code = 478. So codes go sparse SOMEWHERE between those two
checkpoints. This script prints max code after every step so we can
see the exact transition.

Usage:
    python -m mlframe.profiling.trace_category_codes --parquet "R:\\..\\jobs_details.parquet"
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


CULPRIT = "category"
DROP_COLS = [
    "uid", "job_posted_at", "job_status", "cl_id",
    "_raw_countries", "_raw_languages", "_raw_tags",
    "job_post_source", "job_post_device", "job_post_flow_type",
]


def dump(tag: str, df: pl.DataFrame) -> None:
    if CULPRIT not in df.columns:
        print(f"  [{tag}] category dropped", flush=True)
        return
    s = df[CULPRIT]
    try:
        mx = int(s.to_physical().max())
        mn = int(s.to_physical().min())
        nu = s.n_unique()
        dc = s.to_physical().n_unique()
        chk = s.n_chunks()
        print(f"  [{tag}] dtype={s.dtype}, shape_rows={df.height}, "
              f"n_unique={nu}, n_chunks={chk}, codes=[{mn},{mx}], distinct={dc}",
              flush=True)
    except Exception as e:
        print(f"  [{tag}] error inspecting: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()

    print(f"env: polars {pl.__version__}", flush=True)
    print(f"global string cache enabled: {pl.using_string_cache()}", flush=True)

    t0 = time.perf_counter()
    df = pl.read_parquet(args.parquet)
    print(f"\n=== Step 0: parquet loaded in {time.perf_counter()-t0:.1f}s ===", flush=True)
    dump("0-loaded", df)

    df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    dump("1-f64->f32", df)

    df = df.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
    dump("2-utf8->cat", df)

    df = df.sort("job_posted_at")
    dump("3-sorted", df)

    df = df.with_columns([
        pl.col("job_posted_at").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("job_posted_at").dt.day().cast(pl.Int8).alias("day"),
        pl.col("job_posted_at").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("job_posted_at").dt.month().cast(pl.Int8).alias("month"),
    ])
    dump("4-datetime-feats", df)

    target = (df["cl_act_total_hired"].fill_null(0) >= 1).cast(pl.Int8).to_numpy()
    dump("5-target-extracted", df)

    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(to_drop)
    dump("6-dropped-cols", df)

    HIGH_CARD = 300
    to_drop_text = [c for c, dt in df.schema.items()
                    if dt in (pl.Categorical, pl.Utf8) and df[c].n_unique() > HIGH_CARD]
    if to_drop_text:
        df = df.drop(to_drop_text)
    dump(f"7-text-promoted-drops {to_drop_text}", df)

    all_cat = [c for c, dt in df.schema.items() if dt == pl.Categorical]
    drop_other = [c for c in all_cat if c != CULPRIT]
    if drop_other:
        df = df.drop(drop_other)
    dump(f"8-kept-only-culprit+numerics", df)

    if CULPRIT in df.columns and df[CULPRIT].null_count() > 0:
        df = df.with_columns(pl.col(CULPRIT).fill_null("__MISSING__"))
    dump("9-fill-null", df)

    # Time-ordered split.
    n = df.height
    n_test = int(n * 0.10); n_val = int(n * 0.10); n_train = n - n_val - n_test
    train = df[:n_train]
    dump(f"10-train-full[:{n_train}]", train)

    train = train[:211_168]
    dump("11-train-sliced[:211168]", train)

    val = df[n_train : n_train + n_val]
    dump(f"12-val-slice[{n_train}:{n_train+n_val}]", val)

    print("\n=== Summary ===", flush=True)
    print(f"  Find the first step where codes went sparse (max code jumps "
          f"from single/low digits to millions).", flush=True)


if __name__ == "__main__":
    main()
