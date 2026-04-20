r"""Minimal synthetic reproducer for XGBoost 3.2 access violation on
Windows when a pl.Categorical column inherits scattered physical codes
from the polars global StringCache.

Discovered via column bisection on production data 2026-04-20:
  minimum ingredients = ONE high-cardinality string column
  (skills_text, ~2M uniques) + ONE low-cardinality string column
  (category, 89 uniques).

Mechanism (confirmed):
  1. polars 1.19+ has the global StringCache permanently enabled
     (disable_string_cache() is a no-op in 1.35).
  2. Casting both string columns through the shared cache registers
     skills_text's ~2M entries.
  3. fill_null('__MISSING__') on category re-resolves its values
     through the cache → scattered physical codes up to ~2.5M, with
     only 89 distinct.
  4. XGB.fit() with enable_categorical=True reads physical codes
     directly into a bin array sized for 89 categories. Index ~2.5M
     → out-of-bounds → Windows SEH 0xC0000005 / silent kill.

Usage:
    python -m mlframe.profiling.repro_xgb_skills_text_trigger
    python -m mlframe.profiling.repro_xgb_skills_text_trigger --fix-enum

Expected on Windows + xgboost 3.2.0 + polars 1.33+:
    no --fix-enum: silent kill, exit 3221226505 (0xC0000005)
    --fix-enum:    FIT_OK via pl.Enum(union) workaround
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
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=211_168)
    ap.add_argument("--n-val", type=int, default=100_000)
    ap.add_argument("--n-skills-uniques", type=int, default=2_000_000,
                    help="Number of unique strings in skills_text (high-card). "
                         "Prod had 2_063_092.")
    ap.add_argument("--n-category-uniques", type=int, default=89,
                    help="Number of unique strings in category (low-card). "
                         "Prod had 89.")
    ap.add_argument("--n-numeric", type=int, default=95,
                    help="Float32 numeric columns alongside.")
    ap.add_argument("--fix-enum", action="store_true",
                    help="Apply pl.Enum(union) workaround; expected FIT_OK.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import xgboost
    print(f"env: python {sys.version.split()[0]}, polars {pl.__version__}, "
          f"xgboost {xgboost.__version__}, platform {sys.platform}", flush=True)
    print(f"pl.using_string_cache() = {pl.using_string_cache()}", flush=True)

    rng = np.random.default_rng(args.seed)

    # Build the full (unsliced) frame.
    n_full = args.n_train + args.n_val + 100_000  # margin for slicing
    skills_pool = [f"skill_{i:07d}" for i in range(args.n_skills_uniques)]
    category_pool = [f"cat_{i:03d}" for i in range(args.n_category_uniques)]

    t0 = time.perf_counter()
    skills_col = rng.choice(skills_pool, size=n_full).tolist()
    # Force all unique skill values into the sample so cardinality matches.
    # (rng.choice on 2M pool with 400k draws won't visit all entries.)
    # We overwrite first N rows with each unique skill to guarantee coverage.
    for i in range(min(args.n_skills_uniques, n_full)):
        skills_col[i] = skills_pool[i]
    rng.shuffle(skills_col)

    category_col = rng.choice(category_pool, size=n_full).tolist()
    # Inject a few nulls in category so fill_null has something to do.
    null_mask = rng.random(n_full) < 0.01
    category_col = [None if m else v for v, m in zip(category_col, null_mask)]

    df = pl.DataFrame({"skills_text": skills_col, "category": category_col})
    print(f"built raw strings in {time.perf_counter()-t0:.1f}s, shape={df.shape}",
          flush=True)

    # Cast both together — shared cache batch.
    t0 = time.perf_counter()
    df = df.with_columns([
        pl.col("skills_text").cast(pl.Categorical),
        pl.col("category").cast(pl.Categorical),
    ])
    print(f"cast both to Categorical in {time.perf_counter()-t0:.1f}s",
          flush=True)
    mx = df["category"].to_physical().max()
    print(f"  category physical_codes_max after cast: {mx}", flush=True)

    # Critical: fill_null on category. This is the step that re-resolves
    # category's codes through the global StringCache, scattering them.
    before = df["category"].to_physical().max()
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    after = df["category"].to_physical().max()
    print(f"  category physical_codes_max: before fill_null={before}, "
          f"after fill_null={after}", flush=True)

    # Drop skills_text — XGB never sees it. Its only purpose was to
    # pollute the cache during the co-cast step.
    df = df.drop("skills_text")

    if args.fix_enum:
        print("  APPLYING WORKAROUND: cast category to pl.Enum(sorted uniques)",
              flush=True)
        uniques = sorted(df["category"].unique().to_list())
        df = df.with_columns(pl.col("category").cast(pl.Enum(uniques)))
        mx = df["category"].to_physical().max()
        print(f"  category physical_codes_max after enum: {mx}", flush=True)

    # Add numeric features.
    for i in range(args.n_numeric):
        df = df.with_columns(
            pl.Series(f"num_{i}", rng.standard_normal(n_full).astype(np.float32)),
        )

    # Time-ordered split surrogate: just slice sequentially.
    train = df[:args.n_train]
    val   = df[args.n_train : args.n_train + args.n_val]
    y_tr = rng.integers(0, 2, args.n_train).astype(np.int8)
    y_v  = rng.integers(0, 2, args.n_val).astype(np.int8)

    mx = int(train["category"].to_physical().max())
    dc = int(train["category"].to_physical().n_unique())
    nu = train["category"].n_unique()
    print(f"  train[category]: n_unique={nu}, physical_codes_max={mx}, "
          f"distinct_codes={dc}", flush=True)
    print(f"  train.shape={train.shape}, val.shape={val.shape}", flush=True)

    print(f"\nCalling XGBClassifier.fit() — expect silent kill without "
          f"--fix-enum on Windows + xgboost 3.2.0.", flush=True)
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
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s "
              f"(best_iter={model.best_iteration})", flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
