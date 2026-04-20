r"""In-memory reproducer — loads the production parquet, does the same
preprocessing the bisector worker did, and calls XGB.fit without any
intermediate parquet roundtrip. If this crashes but
``D:\Temp\xgb_crash_slice\reproduce_xgb_crash.py`` (bundle) doesn't,
the trigger is something parquet write/read normalizes away —
likely Polars' multi-chunk Categorical state inherited from the
sliced parent frame.

Usage:
    python -m mlframe.profiling.reproduce_xgb_inmemory \
        --parquet "R:\\Data\\Upwork\\dataframes\\PRODUCTION\\jobs_details.parquet"

Expected:
    exit 3221226505 (0xC0000005 access violation), no traceback.

Then try each of the following toggles to narrow the trigger down:

    --rechunk             # force train.rechunk() before fit
    --cast-via-parquet    # train.write_parquet()+read_parquet() before fit
    --to-enum             # cast category to pl.Enum(sorted_uniques)

The first of these to make the crash stop tells us which property
of the bisector's intermediate state is the trigger.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


DROP_COLS = [
    "uid", "job_posted_at", "job_status", "cl_id",
    "_raw_countries", "_raw_languages", "_raw_tags",
    "job_post_source", "job_post_device", "job_post_flow_type",
]
CULPRIT_CATS = ["category"]
ROW_LIMIT = 211_168


def load_and_prep(parquet_path: str, *, rechunk: bool, cast_via_parquet: bool,
                  to_enum: bool):
    print(f"Loading {parquet_path}...", flush=True)
    df = pl.read_parquet(parquet_path)
    df = (
        df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
          .with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
          .sort("job_posted_at")
    )
    df = df.with_columns([
        pl.col("job_posted_at").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("job_posted_at").dt.day().cast(pl.Int8).alias("day"),
        pl.col("job_posted_at").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("job_posted_at").dt.month().cast(pl.Int8).alias("month"),
    ])
    target = (df["cl_act_total_hired"].fill_null(0) >= 1).cast(pl.Int8).to_numpy()

    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(to_drop)

    # Text-promote high-cardinality cats
    HIGH_CARD = 300
    to_drop_text = [c for c, dt in df.schema.items()
                    if dt in (pl.Categorical, pl.Utf8) and df[c].n_unique() > HIGH_CARD]
    if to_drop_text:
        df = df.drop(to_drop_text)

    # Keep only culprit cat + all numerics
    all_cat = [c for c, dt in df.schema.items() if dt == pl.Categorical]
    drop_other = [c for c in all_cat if c not in CULPRIT_CATS]
    df = df.drop(drop_other)

    # Round-17 fill_null on culprit if it has nulls
    for c in CULPRIT_CATS:
        if c in df.columns and df[c].null_count() > 0:
            df = df.with_columns(pl.col(c).fill_null("__MISSING__"))

    # Time-ordered split
    n = df.height
    n_test = int(n * 0.10); n_val = int(n * 0.10); n_train = n - n_val - n_test
    train = df[:n_train]
    val = df[n_train : n_train + n_val]
    y_tr = target[:n_train]
    y_v = target[n_train : n_train + n_val]

    # Slice train to minimum crashing rows
    train = train[:ROW_LIMIT]
    y_tr = y_tr[:ROW_LIMIT]

    # --- Toggles to narrow the trigger ---

    for c in CULPRIT_CATS:
        if c in train.columns:
            print(f"  train[{c}]: n_chunks={train[c].n_chunks()}, "
                  f"n_unique={train[c].n_unique()}, "
                  f"physical_codes_range=[{train[c].to_physical().min()}, "
                  f"{train[c].to_physical().max()}], "
                  f"distinct_codes={train[c].to_physical().n_unique()}",
                  flush=True)

    if rechunk:
        print("  TOGGLE: train.rechunk()", flush=True)
        train = train.rechunk()
        val = val.rechunk()
        for c in CULPRIT_CATS:
            print(f"  after rechunk: train[{c}].n_chunks={train[c].n_chunks()}",
                  flush=True)

    if cast_via_parquet:
        print("  TOGGLE: roundtrip through parquet", flush=True)
        tmp = Path("_inmem_rt_train.parquet"); tmpv = Path("_inmem_rt_val.parquet")
        train.write_parquet(tmp); val.write_parquet(tmpv)
        train = pl.read_parquet(tmp); val = pl.read_parquet(tmpv)
        # Windows keeps a file lock after read; swallow cleanup errors
        # so we still get to the fit. Leftover files next to cwd are harmless.
        for p in (tmp, tmpv):
            try:
                p.unlink()
            except (OSError, PermissionError) as _e:
                print(f"  (cleanup skipped for {p.name}: {_e})", flush=True)
        for c in CULPRIT_CATS:
            print(f"  after parquet rt: train[{c}].n_chunks={train[c].n_chunks()}, "
                  f"n_unique={train[c].n_unique()}, "
                  f"physical_codes_range=[{train[c].to_physical().min()}, "
                  f"{train[c].to_physical().max()}]",
                  flush=True)

    if to_enum:
        print("  TOGGLE: cast culprit cats to pl.Enum(union)", flush=True)
        for c in CULPRIT_CATS:
            tr_u = set(train[c].drop_nulls().unique().to_list())
            v_u  = set(val[c].drop_nulls().unique().to_list())
            union = sorted(tr_u | v_u)
            dt = pl.Enum(union)
            train = train.with_columns(pl.col(c).cast(dt))
            val   = val.with_columns(pl.col(c).cast(dt))

    return train, val, y_tr, y_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=str)
    ap.add_argument("--rechunk", action="store_true")
    ap.add_argument("--cast-via-parquet", action="store_true")
    ap.add_argument("--to-enum", action="store_true")
    args = ap.parse_args()

    train, val, y_tr, y_v = load_and_prep(
        args.parquet, rechunk=args.rechunk,
        cast_via_parquet=args.cast_via_parquet, to_enum=args.to_enum,
    )
    print(f"train shape={train.shape}, val shape={val.shape}", flush=True)
    print(f"train dtypes: {dict(train.schema)}", flush=True)

    print("Calling XGB.fit() — expect silent kill unless a toggle fixed it.",
          flush=True)
    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1,
        max_cat_to_onehot=1, max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    try:
        model.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s — bug did NOT reproduce",
              flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
