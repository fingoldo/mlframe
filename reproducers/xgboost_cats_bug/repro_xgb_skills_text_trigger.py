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

    # Mimic prod sequence EXACTLY:
    #   (1) parquet-load: category arrives as Categorical with a
    #       per-Series dict (compact codes 0..N). Simulated by casting
    #       category first, BEFORE the cache has any skills entries.
    #   (2) ``.cast(pl.Utf8).cast(pl.Categorical)`` over other string
    #       columns: skills_text gets 2M values added to the global
    #       cache. category is already Categorical — no recast here.
    #   (3) fill_null('__MISSING__') on category: this is the step
    #       that re-resolves category's values through the NOW-polluted
    #       cache. Compact codes -> scattered. THIS is the trigger
    #       sequence we need to reproduce.

    # STEP 1: Interleaved cache pollution.
    # To mimic prod's parquet-write state (cache has been populated by
    # many string columns in a specific interleaved order), we cast a
    # Series that contains ~N skill strings + the category strings
    # interleaved at ~even intervals. This assigns each category
    # string a SCATTERED physical code somewhere inside the skills
    # code range, instead of a contiguous block at the end.
    t0 = time.perf_counter()
    n_skills = args.n_skills_uniques
    n_cats = args.n_category_uniques
    category_pool = [f"cat_{i:03d}" for i in range(n_cats)]
    skills_pool = [f"skill_{i:07d}" for i in range(n_skills)]
    # Build mixed list: after every ``step`` skills, inject one category.
    step = max(1, n_skills // (n_cats + 1))
    mixed = []
    cat_idx = 0
    for i, s in enumerate(skills_pool):
        mixed.append(s)
        if i > 0 and i % step == 0 and cat_idx < n_cats:
            mixed.append(category_pool[cat_idx])
            cat_idx += 1
    # Any categories we didn't inject go at the tail.
    while cat_idx < n_cats:
        mixed.append(category_pool[cat_idx])
        cat_idx += 1
    _ = pl.Series("cache_primer", mixed, dtype=pl.Categorical)
    print(f"step 1 (interleaved cache pollution: "
          f"{n_skills:_} skills + {n_cats} cats): {time.perf_counter()-t0:.1f}s",
          flush=True)
    # Verify scatter: each category string should now have a scattered
    # physical code in [step, 2*step, 3*step, ...].
    probe = pl.Series("probe", category_pool, dtype=pl.Categorical)
    probe_codes = sorted(probe.to_physical().to_list())
    print(f"  category values now have codes: "
          f"first3={probe_codes[:3]}, last3={probe_codes[-3:]}, "
          f"max={probe_codes[-1]}", flush=True)

    # STEP 2: Build the category Series. Values are already in the
    # cache at scattered codes, so the Series inherits those codes.
    t0 = time.perf_counter()
    n_full = args.n_train + args.n_val + 100_000
    category_col = rng.choice(category_pool, size=n_full).tolist()
    null_mask = rng.random(n_full) < 0.01
    category_col = [None if m else v for v, m in zip(category_col, null_mask)]
    category_series = pl.Series("category", category_col, dtype=pl.Categorical)
    cc = category_series.to_physical().drop_nulls().unique().to_list()
    print(f"step 2 (build category Series): max code "
          f"{max(cc) if cc else 'empty'}, distinct {len(cc)}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # STEP 3a: Wrap into DataFrame and prepare for fill_null.
    df = pl.DataFrame({"category": category_series})
    before = df["category"].to_physical().drop_nulls().max()
    print(f"step 3a (before fill_null): category max code = {before}",
          flush=True)

    # STEP 3b: fill_null — the trigger.
    before = df["category"].to_physical().max()
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    after = df["category"].to_physical().max()
    print(f"step 3b (fill_null): category max code "
          f"{before} -> {after}  "
          f"({'JUMP' if after > before * 2 else 'no-jump'})", flush=True)

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
