r"""Dump a minimal raw (no anonymisation) crash bundle for the XGBoost upstream issue.

The key insight missing from earlier dump scripts:
  - polars parquet round-trip resets physical Categorical codes to [0..n_unique-1].
    The sparse code 2_526_058 that causes crash exists only in the in-memory StringCache.
  - dump_xgb_prodcrash_slice.py drops skills_text (high-cardinality), removing the
    cache-pollution mechanism entirely. Its bundle cannot crash on reproduce.

This script keeps skills_text as raw String so reproduce.py can replicate
the exact cast sequence that generates the sparse code.

Bundle layout
-------------
  crash_data.parquet   — all rows needed, columns: skills_text (String) +
                         category (String) + cl_act_total_hired (numeric target)
  reproduce.py         — standalone: loads parquet, replicates cast order,
                         calls XGBClassifier.fit; expected to crash on Windows

Usage (on prod box)
-------------------
    python dump_raw_crash_bundle.py --parquet "R:\Data\Upwork\dataframes\PRODUCTION\jobs_details.parquet" --out-dir D:\Temp\xgb_raw_bundle

Then copy D:\Temp\xgb_raw_bundle\ here and run:
    python reproduce.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

TIMESTAMP_COLUMN = "job_posted_at"
TARGET_COLUMN    = "cl_act_total_hired"
TARGET_THRESHOLD = 1
MIN_ROWS_TRAIN   = 211_168   # bisect-confirmed threshold on this machine
N_VAL            = 100_000


REPRO_SCRIPT = '''"""Standalone reproducer: XGBoost 3.2.0 + polars Categorical sparse-code crash.

Steps that trigger the bug:
  1. Cast skills_text (String, ~2.5M uniques) -> Categorical.
     Fills global StringCache with ~2.5M entries.
  2. Cast category (String, 89 uniques) -> Categorical through the polluted cache.
     Its 89 strings get scattered codes in [0..88] range from the dict order.
  3. fill_null("__MISSING__") on category.
     "__MISSING__" is registered in the polluted cache at code ~2_526_059.
     category physical-code max jumps from ~88 to ~2_526_059.
  4. XGBClassifier.fit() with enable_categorical=True, tree_method="hist".
     XGB reads max_cat=2_526_059, allocates cut_values[2_526_060] (~10 MB).
     On Windows this oversize allocation corrupts the heap -> 0xC0000005.

Expected: silent process exit with rc=3221226505 (0xC0000005).
Workaround: cast category to pl.Enum(sorted_uniques) after fill_null.
"""
from __future__ import annotations

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
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

HERE = Path(__file__).parent
WORKAROUND = "--workaround" in sys.argv


def main():
    import xgboost
    print(f"polars {pl.__version__}, xgboost {xgboost.__version__}, "
          f"platform={sys.platform}, using_string_cache={pl.using_string_cache()}",
          flush=True)

    t0 = time.perf_counter()
    df = pl.read_parquet(HERE / "crash_data.parquet")
    print(f"loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  skills_text n_unique={df['skills_text'].n_unique():_}, "
          f"category n_unique={df['category'].n_unique()}", flush=True)

    # Step 1: cast skills_text FIRST to pollute the StringCache.
    df = df.with_columns(pl.col("skills_text").cast(pl.Categorical))
    print(f"  after skills_text cast: cache size ~ {df['skills_text'].n_unique():_}",
          flush=True)

    # Step 2: cast category through the now-polluted cache.
    df = df.with_columns(pl.col("category").cast(pl.String).cast(pl.Categorical))
    print(f"  after category cast: "
          f"n_unique={df['category'].n_unique()}, "
          f"codes_max={df['category'].to_physical().drop_nulls().max()}",
          flush=True)

    # Step 3: fill_null registers __MISSING__ at a very high cache code.
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    print(f"  after fill_null: "
          f"n_unique={df['category'].n_unique()}, "
          f"codes_max={df['category'].to_physical().max()}",
          flush=True)

    if WORKAROUND:
        print("  WORKAROUND: casting category to pl.Enum(sorted_uniques)", flush=True)
        uniques = sorted(df["category"].unique().drop_nulls().to_list())
        df = df.with_columns(pl.col("category").cast(pl.Enum(uniques)))
        print(f"  codes_max after Enum cast: {df['category'].to_physical().max()}",
              flush=True)

    # Drop skills_text — XGB only receives category.
    df = df.drop("skills_text")

    n_train = __import__("builtins").min(211_168, int(df.height * 0.8))
    n_val   = __import__("builtins").min(100_000, df.height - n_train)
    train = df[:n_train]
    val   = df[n_train : n_train + n_val]

    rng = np.random.default_rng(42)
    y_tr = rng.integers(0, 2, n_train, dtype=np.int8)
    y_v  = rng.integers(0, 2, n_val,   dtype=np.int8)

    print(f"\\nfitting XGB on train={train.shape} val={val.shape} -- "
          f"expect silent kill (exit 3221226505) on Windows",
          flush=True)
    m = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1,
        max_cat_to_onehot=1, max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    t0 = time.perf_counter()
    try:
        m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s -- bug did NOT reproduce",
              flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True,
                    help="Path to jobs_details.parquet on prod")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (created if missing)")
    ap.add_argument("--n-train", type=int, default=MIN_ROWS_TRAIN,
                    help=f"Train rows (default: {MIN_ROWS_TRAIN} = bisect threshold)")
    ap.add_argument("--n-val", type=int, default=N_VAL,
                    help=f"Val rows (default: {N_VAL})")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} ...", flush=True)
    df = pl.read_parquet(args.parquet)
    print(f"  loaded {df.shape}", flush=True)

    # Sort by timestamp to match original pipeline ordering.
    if TIMESTAMP_COLUMN in df.columns:
        df = df.sort(TIMESTAMP_COLUMN)

    # Keep only the two columns that matter for the crash + enough rows.
    needed = args.n_train + args.n_val
    if df.height < needed:
        print(f"WARNING: parquet has only {df.height:,} rows, need {needed:,}. "
              f"Using all rows.", flush=True)
        needed = df.height

    df = df[:needed]

    # Retain skills_text as String (NOT cast to Categorical — that happens
    # in reproduce.py to replicate the exact cache-pollution sequence).
    keep = []
    if "skills_text" in df.columns:
        keep.append(pl.col("skills_text").cast(pl.String))
    else:
        sys.exit("ERROR: 'skills_text' column not found in parquet.")

    if "category" in df.columns:
        keep.append(pl.col("category").cast(pl.String))
    else:
        sys.exit("ERROR: 'category' column not found in parquet.")

    df = df.select(keep)

    print(f"  slice: {df.shape}", flush=True)
    print(f"  skills_text n_unique={df['skills_text'].n_unique():_}", flush=True)
    print(f"  category n_unique={df['category'].n_unique()}, "
          f"nulls={df['category'].null_count()}", flush=True)

    out_parquet = out_dir / "crash_data.parquet"
    df.write_parquet(out_parquet)

    out_repro = out_dir / "reproduce.py"
    out_repro.write_text(REPRO_SCRIPT, encoding="utf-8")

    sz_parquet = out_parquet.stat().st_size / 1e6
    print(f"\nBundle written to {out_dir}/", flush=True)
    print(f"  crash_data.parquet : {sz_parquet:.1f} MB", flush=True)
    print(f"  reproduce.py", flush=True)
    print(f"\nVerify crash:", flush=True)
    print(f"    cd {out_dir}", flush=True)
    print(f"    python reproduce.py", flush=True)
    print(f"Expected: silent exit 3221226505 (0xC0000005).", flush=True)
    print(f"\nVerify workaround:", flush=True)
    print(f"    python reproduce.py --workaround", flush=True)
    print(f"Expected: FIT_OK.", flush=True)


if __name__ == "__main__":
    main()
