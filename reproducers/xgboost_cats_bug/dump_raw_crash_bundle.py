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

    # Step 1: load ALL unique skills_text values and cast to Categorical.
    # This primes the StringCache with ~2.5M entries — the cache-pollution
    # step that makes subsequent category codes land at sparse positions.
    skills_path = HERE / "skills_text_uniques.parquet"
    if not skills_path.exists():
        # fallback for old bundles that have crash_data.parquet with skills_text
        skills_path = HERE / "crash_data.parquet"
    t0 = time.perf_counter()
    skills = pl.read_parquet(skills_path, columns=["skills_text"])
    print(f"priming StringCache with {skills.height:_} unique skills_text values...",
          flush=True)
    _ = skills["skills_text"].cast(pl.Categorical)
    print(f"  done in {time.perf_counter()-t0:.1f}s, "
          f"using_string_cache={pl.using_string_cache()}", flush=True)

    # Step 2: load the crash slice (category only) and cast through polluted cache.
    slice_path = HERE / "crash_slice.parquet"
    if not slice_path.exists():
        slice_path = HERE / "crash_data.parquet"
    t0 = time.perf_counter()
    df = pl.read_parquet(slice_path)
    print(f"loaded crash_slice {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  category n_unique={df['category'].n_unique()}", flush=True)

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
    if "skills_text" not in pl.read_parquet(args.parquet, n_rows=1).columns:
        sys.exit("ERROR: 'skills_text' column not found in parquet.")
    if "category" not in pl.read_parquet(args.parquet, n_rows=1).columns:
        sys.exit("ERROR: 'category' column not found in parquet.")

    df = pl.read_parquet(args.parquet, columns=["skills_text", "category",
                                                 TIMESTAMP_COLUMN])
    print(f"  loaded {df.shape}", flush=True)

    # Sort by timestamp to match original pipeline ordering.
    if TIMESTAMP_COLUMN in df.columns:
        df = df.sort(TIMESTAMP_COLUMN).drop(TIMESTAMP_COLUMN)

    # --- skills_text_uniques: ALL unique values from full dataset -----------
    # The cache-pollution mechanism requires ~2.5M unique strings to be cast
    # to Categorical before category is resolved. We deduplicate across the
    # full dataset so reproduce.py can prime the StringCache correctly even
    # though it only fits XGB on 311k rows.
    skills_uniques = (
        df["skills_text"]
        .drop_nulls()
        .unique()
        .cast(pl.String)
        .to_frame("skills_text")
    )
    print(f"  skills_text n_unique={skills_uniques.height:_} "
          f"(from full {df.height:_}-row dataset)", flush=True)

    # --- crash_slice: only the rows needed for the XGB fit -----------------
    needed = args.n_train + args.n_val
    if df.height < needed:
        print(f"WARNING: parquet has only {df.height:,} rows, need {needed:,}. "
              f"Using all rows.", flush=True)
        needed = df.height

    slice_df = df[:needed].select(
        pl.col("category").cast(pl.String)
    )
    print(f"  crash_slice: {slice_df.shape}, "
          f"category n_unique={slice_df['category'].n_unique()}, "
          f"nulls={slice_df['category'].null_count()}", flush=True)

    out_skills  = out_dir / "skills_text_uniques.parquet"
    out_slice   = out_dir / "crash_slice.parquet"
    skills_uniques.write_parquet(out_skills)
    slice_df.write_parquet(out_slice)

    out_repro = out_dir / "reproduce.py"
    out_repro.write_text(REPRO_SCRIPT, encoding="utf-8")

    print(f"\nBundle written to {out_dir}/", flush=True)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB", flush=True)
    print(f"\nVerify crash:", flush=True)
    print(f"    cd {out_dir}", flush=True)
    print(f"    python reproduce.py", flush=True)
    print(f"Expected: silent exit 3221226505 (0xC0000005).", flush=True)
    print(f"\nVerify workaround:", flush=True)
    print(f"    python reproduce.py --workaround", flush=True)
    print(f"Expected: FIT_OK.", flush=True)


if __name__ == "__main__":
    main()
