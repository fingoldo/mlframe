r"""Dump the minimal crashing slice identified by
``bisect_xgb_prodcrash.py`` to a self-contained parquet + provide a
companion standalone reproducer that loads it and triggers the
XGBoost 3.2.0 access violation on Windows.

Bisection result (2026-04-20 on prod_jobsdetails, 128 GB Windows):
  culprit cat set:    ['category']       — pl.Categorical, 89 uniques
  min rows to crash:  211_168
  crash signature:    Windows SEH exit 0xC0000005 (access violation),
                      no Python traceback, no C++ exception surface

Usage
-----

1. Run once on the prod box to dump the anonymised slice:

       python -m mlframe.profiling.dump_xgb_prodcrash_slice \
           --parquet "R:\\Data\\Upwork\\dataframes\\PRODUCTION\\jobs_details.parquet" \
           --out-dir D:\\Temp\\xgb_crash_slice

   Produces in the out dir:
     - ``crash_train.parquet``  (211_168 rows, ~100 cols)
     - ``crash_val.parquet``    (~720 k rows, same cols)
     - ``crash_target_train.npy`` / ``crash_target_val.npy``
     - ``reproduce_xgb_crash.py``  — standalone script that loads
       the parquets and calls XGBClassifier.fit, expected to
       silently kill the process.

2. Verify the standalone script actually crashes:

       cd D:\\Temp\\xgb_crash_slice
       python reproduce_xgb_crash.py

   Expected on Windows: exit code 3221226505 (0xC0000005), no
   traceback. If so, the bundle is upstream-submissable: attach the
   4 small files + the reproducer script to an xgboost GitHub issue.

Anonymisation: column NAMES are preserved (they're part of the repro
shape). ``category`` string values are replaced with opaque
``c_000..c_088`` tokens to avoid leaking business data.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import polars as pl

# Constants from bisection — kept in sync with the original
# ``bisect_xgb_prodcrash.py`` logic.
DEFAULT_DROP_COLUMNS = [
    "uid", "job_posted_at", "job_status", "cl_id",
    "_raw_countries", "_raw_languages", "_raw_tags",
    "job_post_source", "job_post_device", "job_post_flow_type",
]
CULPRIT_CATS = ["category"]
MIN_ROWS_TO_CRASH = 211_168

TARGET_COLUMN = "cl_act_total_hired"
TARGET_THRESHOLD = 1
TIMESTAMP_COLUMN = "job_posted_at"
HIGH_CARD_TEXT_THRESHOLD = 300


def anonymise_category(s: pl.Series) -> pl.Series:
    """Replace string values in a pl.Categorical with ``c_NNN`` tokens
    preserving cardinality, order, and null positions exactly.

    This lets us publish the crash slice without leaking business
    category names while keeping every bit that affects XGB's hist
    path (cardinality, frequencies, null mask, physical code layout)
    identical.
    """
    # Preserve physical codes (order-of-first-occurrence) by building
    # a map from original strings to synthetic tokens.
    uniq = s.drop_nulls().unique().to_list()
    mapping = {orig: f"c_{i:03d}" for i, orig in enumerate(uniq)}
    renamed = s.cast(pl.String).replace_strict(
        list(mapping.keys()), list(mapping.values()), default=None,
    ).cast(pl.Categorical)
    return renamed


def build_crash_slice(parquet_path: str) -> Tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """Replicate the bisector's preprocessing and slice to the minimal
    crashing shape: ``ROW_LIMIT`` rows of train + full val, ``category``
    only kept among categoricals, target column included and
    fill_null(0) applied to avoid NaN propagation.
    """
    print(f"Loading {parquet_path}...", flush=True)
    df = pl.read_parquet(parquet_path)
    df = (
        df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
          .with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
          .sort(TIMESTAMP_COLUMN)
    )
    df = df.with_columns([
        pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Int8).alias("hour"),
        pl.col(TIMESTAMP_COLUMN).dt.day().cast(pl.Int8).alias("day"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col(TIMESTAMP_COLUMN).dt.month().cast(pl.Int8).alias("month"),
    ])
    # Target with fill_null(0) — nulls treated as below-threshold class.
    target_series = df[TARGET_COLUMN].fill_null(0) >= TARGET_THRESHOLD
    target = target_series.cast(pl.Int8).to_numpy()

    # Drop extractor-requested + drift-suspect columns.
    to_drop = [c for c in DEFAULT_DROP_COLUMNS if c in df.columns]
    df = df.drop(to_drop)

    # Text-promote high-cardinality cats.
    to_drop_text = [
        c for c, dt in df.schema.items()
        if dt in (pl.Categorical, pl.Utf8) and df[c].n_unique() > HIGH_CARD_TEXT_THRESHOLD
    ]
    if to_drop_text:
        df = df.drop(to_drop_text)
        print(f"  dropped text-promoted columns: {to_drop_text}", flush=True)

    # Keep only the culprit cat + all numerics.
    all_cat = [c for c, dt in df.schema.items() if dt == pl.Categorical]
    drop_other_cat = [c for c in all_cat if c not in CULPRIT_CATS]
    df = df.drop(drop_other_cat)
    print(f"  kept cats: {CULPRIT_CATS}, dropped: {drop_other_cat}", flush=True)

    # Anonymise the culprit column.
    for c in CULPRIT_CATS:
        if c in df.columns:
            df = df.with_columns(anonymise_category(df[c]).alias(c))
    # Apply Round-17 fill_null('__MISSING__') on the culprit so the
    # standalone repro matches exactly what XGB saw when it crashed.
    for c in CULPRIT_CATS:
        if c in df.columns and df[c].null_count() > 0:
            df = df.with_columns(pl.col(c).fill_null("__MISSING__"))

    # Time-ordered 80/10/10 split, matching the bisector.
    n = df.height
    n_test = int(n * 0.10)
    n_val = int(n * 0.10)
    n_train = n - n_val - n_test
    train_full = df[:n_train]
    val_df     = df[n_train : n_train + n_val]
    y_train_full = target[:n_train]
    y_val        = target[n_train : n_train + n_val]

    # Trim train to the minimal crashing row count.
    train_df = train_full[:MIN_ROWS_TO_CRASH]
    y_train = y_train_full[:MIN_ROWS_TO_CRASH]
    print(f"  train sliced to {train_df.shape}, val {val_df.shape}", flush=True)

    return train_df, val_df, y_train, y_val


REPRO_SCRIPT = r'''"""Minimal standalone reproducer: XGBoost 3.2 access violation on
Windows with a single pl.Categorical column at ~200k rows.

Bundle contents (must be in the same directory as this script):
  crash_train.parquet          — 211_168 rows x ~100 cols, sorted
                                  by original timestamp
  crash_val.parquet            — ~720k rows, same schema
  crash_target_train.npy       — int8 binary target, length 211_168
  crash_target_val.npy         — int8 binary target, length ~720k

Expected behavior on Windows + XGBoost 3.2.0:
  Process silently exits with code 3221226505 (0xC0000005,
  STATUS_ACCESS_VIOLATION). No Python traceback.

Expected on Linux or with xgboost.__version__ != 3.2.*:
  Behaviour not verified — may or may not reproduce.

Workaround:
  Before ``.fit()``, cast the Polars Categorical columns to a
  ``pl.Enum(union_of_values)`` shared across train and val. XGB
  then takes a different internal path and fits normally.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

# Show native crashes instead of silently dying on Windows.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # suppress WER modal
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass


def main():
    here = Path(__file__).parent
    print("Loading bundled parquet...", flush=True)
    train = pl.read_parquet(here / "crash_train.parquet")
    val   = pl.read_parquet(here / "crash_val.parquet")
    y_tr = np.load(here / "crash_target_train.npy")
    y_v  = np.load(here / "crash_target_val.npy")

    print(f"  train shape={train.shape}, val shape={val.shape}", flush=True)
    print(f"  train categorical cols: "
          f"{[c for c,dt in train.schema.items() if dt == pl.Categorical]}",
          flush=True)
    # Confirm the culprit column state:
    for c in [c for c,dt in train.schema.items() if dt == pl.Categorical]:
        print(f"  {c}: n_unique={train[c].n_unique()}, "
              f"nulls={train[c].null_count()}", flush=True)

    print("Calling XGBClassifier.fit() — expect silent kill "
          "(exit 3221226505) on Windows + xgboost 3.2.0.", flush=True)
    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=5,
        enable_categorical=True,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        verbosity=1,
        max_cat_to_onehot=1,
        max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
    print(f"FIT_OK in {time.perf_counter()-t0:.1f}s "
          f"(BUG DID NOT REPRODUCE — unexpected)", flush=True)


if __name__ == "__main__":
    main()
'''


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet", required=True, type=str,
                    help="Path to jobs_details.parquet")
    ap.add_argument("--out-dir", required=True, type=str,
                    help="Directory to write the bundle into (created if missing)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, y_train, y_val = build_crash_slice(args.parquet)

    print(f"\nWriting bundle to {out_dir}...", flush=True)
    train_df.write_parquet(out_dir / "crash_train.parquet")
    val_df.write_parquet(out_dir / "crash_val.parquet")
    np.save(out_dir / "crash_target_train.npy", y_train)
    np.save(out_dir / "crash_target_val.npy", y_val)
    (out_dir / "reproduce_xgb_crash.py").write_text(REPRO_SCRIPT, encoding="utf-8")

    # Size report — upstream issue attachments have limits.
    total = 0
    for f in out_dir.iterdir():
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name}: {sz/1e6:.2f} MB", flush=True)
    print(f"Total bundle: {total/1e6:.2f} MB", flush=True)

    print(f"\nNow verify the reproducer actually crashes:", flush=True)
    print(f"    cd {out_dir}", flush=True)
    print(f"    python reproduce_xgb_crash.py", flush=True)
    print(f"\nExpected: exit code 3221226505 (0xC0000005), no traceback.",
          flush=True)


if __name__ == "__main__":
    main()
