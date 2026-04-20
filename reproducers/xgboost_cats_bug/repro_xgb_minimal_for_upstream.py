r"""Minimal reproducer for XGBoost 3.2 access violation on Windows.
Bisected from production data 2026-04-20 — smallest known trigger.

Only two columns needed from the source parquet:
  * ``skills_text``  (String, ~2M unique values — cache-pollution source)
  * ``category``     (Categorical, 89 unique values — the crashing column)

No numeric columns are required. No other categoricals. No datetime.
Just those two columns, the fill_null step, and the first 211_168 rows
of the sorted frame.

Environment
-----------
  Windows 10 Pro 10.0.19045, 128 GB RAM, 237 GB pagefile
  Python 3.11.5, polars 1.33.1, xgboost 3.2.0

Expected behaviour on Windows + xgboost 3.2.0:
  Process silently exits with code 3221226505 (0xC0000005,
  STATUS_ACCESS_VIOLATION). No Python traceback. No C++ exception.

Mechanism (empirically isolated via step-by-step polars tracing):
  1. polars 1.19+ keeps the global StringCache permanently enabled
     (``disable_string_cache()`` is a no-op in 1.33+).
  2. Casting ``skills_text`` through ``String -> Categorical`` in the
     same ``with_columns()`` as ``category`` registers ~2M entries in
     the global cache.
  3. ``fill_null('__MISSING__')`` on ``category`` re-resolves the
     column's values through the now-polluted cache. Category's 89
     string values get scattered physical codes spread across the
     [~100, ~3_287_945] range instead of the compact
     [0..88] they had immediately after parquet load.
  4. XGB ``fit()`` with ``enable_categorical=True, tree_method='hist'``
     reads the physical codes directly into a bin array sized for
     ``n_unique`` categories. Indexing that array with scattered
     codes up to 3.3M causes an out-of-bounds access → Windows SEH
     0xC0000005.

Workaround (validated in-process and in production): cast the
``category`` column to ``pl.Enum(sorted(unique_values))`` after
``fill_null``. ``pl.Enum`` enforces compact physical codes by
construction — the bin indexing is well-formed and XGB fits
normally. See ``--workaround``.

Usage
-----
    python repro_xgb_minimal_for_upstream.py --parquet path\\to\\your.parquet

    # Expected: silent process exit with rc=3221226505 (Windows SEH).

    python repro_xgb_minimal_for_upstream.py --parquet path\\to\\your.parquet --workaround

    # Expected: fit completes normally.

The parquet needs:
  * A column ``skills_text`` of dtype ``pl.Utf8`` with very high
    cardinality (~2M+ unique values across the full frame).
  * A column ``category`` of dtype ``pl.Categorical`` (or String)
    with moderate cardinality (~80-100 uniques) and some null rows.
"""
from __future__ import annotations

import argparse
import sys
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="Path to the source parquet (must contain "
                         "'skills_text' + 'category').")
    ap.add_argument("--workaround", action="store_true",
                    help="Cast category to pl.Enum(union) after fill_null. "
                         "Expected to make fit complete normally.")
    ap.add_argument("--n-train", type=int, default=211_168)
    ap.add_argument("--n-val",   type=int, default=100_000)
    args = ap.parse_args()

    print(f"polars={pl.__version__}, xgboost={__import__('xgboost').__version__}, "
          f"platform={sys.platform}, using_string_cache={pl.using_string_cache()}",
          flush=True)

    print(f"loading {args.parquet} ...", flush=True)
    t0 = time.perf_counter()
    df = pl.read_parquet(args.parquet, columns=["skills_text", "category"])
    print(f"  loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Cast skills_text through String to force the shared-cache batch.
    # category may already be Categorical in parquet — route both through
    # String so both participate in the same cast operation.
    df = df.with_columns([
        pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
        pl.col("category").cast(pl.String).cast(pl.Categorical),
    ])
    print(f"  after cast: category n_unique={df['category'].n_unique()}, "
          f"physical_codes_max={df['category'].to_physical().drop_nulls().max()}",
          flush=True)

    # Drop skills_text — XGB only receives category.
    df = df.drop("skills_text")

    # fill_null — the trigger that re-resolves category through the polluted cache.
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    print(f"  after fill_null: category n_unique={df['category'].n_unique()}, "
          f"physical_codes_max={df['category'].to_physical().max()}",
          flush=True)

    if args.workaround:
        print("  APPLYING WORKAROUND: cast to pl.Enum(sorted_uniques)", flush=True)
        uniques = sorted(df["category"].unique().to_list())
        df = df.with_columns(pl.col("category").cast(pl.Enum(uniques)))
        print(f"  after Enum cast: physical_codes_max="
              f"{df['category'].to_physical().max()}", flush=True)

    # Slice first N_TRAIN + N_VAL rows.
    n_needed = args.n_train + args.n_val
    if df.height < n_needed:
        raise SystemExit(f"parquet only has {df.height:,} rows; need {n_needed:,}")
    train = df[:args.n_train]
    val   = df[args.n_train : args.n_train + args.n_val]

    # Random binary target.
    rng = np.random.default_rng(42)
    y_tr = rng.integers(0, 2, size=args.n_train).astype(np.int8)
    y_v  = rng.integers(0, 2, size=args.n_val).astype(np.int8)

    print(f"\nCalling XGBClassifier.fit()...", flush=True)
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
    t0 = time.perf_counter()
    try:
        model.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s  best_iter={model.best_iteration}",
              flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
