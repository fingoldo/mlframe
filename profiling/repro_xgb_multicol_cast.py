r"""Pure synthetic reproducer v3: simulate user's actual polars workflow
of casting MULTIPLE string columns to pl.Categorical simultaneously.

Previous synthetics built a single Categorical with padding. The
user's real pipeline does
  ``.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))``
which casts all string columns in parallel — the cache fills with
values from different columns interspersed, producing scattered
physical codes that differ per column.

This version builds a frame with 3 big string columns (2M, 488k,
113k unique values each — matching prod skills_text/ontology_skills_text
/_raw_segmentation cardinalities) and ONE small string column
(``category``, 89 unique). Casts them all in one ``.with_columns()``.
Then drops the big columns, slices to 211k rows, fits XGB.

If this crashes locally, we have a self-contained upstream repro.
If still FIT_OK, the trigger needs yet more ingredients from prod.
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
    ap.add_argument("--n-rows", type=int, default=500_000,
                    help="Full (parent) frame size. Default 500k — big enough "
                         "to put 2M+ uniques into cache across columns.")
    ap.add_argument("--n-train-slice", type=int, default=211_168)
    ap.add_argument("--n-val-slice", type=int, default=100_000)
    ap.add_argument("--n-numeric", type=int, default=95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=("categorical", "enum"), default="categorical")
    args = ap.parse_args()

    import xgboost, polars
    print(f"env: python {sys.version.split()[0]}, polars {polars.__version__}, "
          f"xgboost {xgboost.__version__}, platform {sys.platform}", flush=True)
    print(f"=== n_rows={args.n_rows:_}, n_numeric={args.n_numeric}, "
          f"mode={args.mode} ===", flush=True)

    rng = np.random.default_rng(args.seed)

    # Big string columns matching prod cardinalities.
    # We can't actually allocate 2M unique strings for 500k rows — so we use
    # smaller pools per column but enough to inflate the cache significantly.
    # Each "big" column still contributes many uniques to the cache.
    t0 = time.perf_counter()
    big_pools = [
        [f"skills_v{i:07d}"   for i in range(200_000)],
        [f"ontolog_v{i:06d}"  for i in range(100_000)],
        [f"segment_v{i:05d}"  for i in range(30_000)],
    ]
    data = {}
    for i, pool in enumerate(big_pools):
        data[f"big_{i}"] = rng.choice(pool, size=args.n_rows).tolist()
    used = [f"cat_{i:03d}" for i in range(89)]  # matches prod `category` cardinality
    data["category"] = rng.choice(used, size=args.n_rows).tolist()
    for j in range(args.n_numeric):
        data[f"num_{j}"] = rng.standard_normal(args.n_rows).astype(np.float32)
    df = pl.DataFrame(data)
    print(f"  built raw frame in {time.perf_counter()-t0:.1f}s, shape={df.shape}", flush=True)

    # Mimic user's prod pipeline: cast ALL string columns simultaneously.
    t0 = time.perf_counter()
    df = df.with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
    print(f"  cast(pl.Categorical) over all Utf8 in {time.perf_counter()-t0:.1f}s", flush=True)

    # Check physical codes on category AFTER the multi-column cast.
    mn = df["category"].to_physical().min()
    mx = df["category"].to_physical().max()
    dc = df["category"].to_physical().n_unique()
    print(f"  category after cast: n_unique={df['category'].n_unique()}, "
          f"physical_codes_range=[{mn}, {mx}], distinct_codes={dc}",
          flush=True)

    # Drop the big columns — the user does similar (drops skills_text etc. as
    # "text features" before XGB).
    df = df.drop([c for c in df.columns if c.startswith("big_")])

    # Slice.
    train = df[:args.n_train_slice]
    val = df[args.n_train_slice : args.n_train_slice + args.n_val_slice]

    # Re-check codes in the slice.
    mn = train["category"].to_physical().min()
    mx = train["category"].to_physical().max()
    dc = train["category"].to_physical().n_unique()
    nu = train["category"].n_unique()
    print(f"  train[category] in slice: n_unique={nu}, "
          f"physical_codes_range=[{mn}, {mx}], distinct_codes={dc}",
          flush=True)

    if args.mode == "enum":
        print("  applying --mode enum workaround (pl.Enum(union))...", flush=True)
        tr_u = set(train["category"].drop_nulls().unique().to_list())
        v_u = set(val["category"].drop_nulls().unique().to_list())
        union = sorted(tr_u | v_u)
        dt = pl.Enum(union)
        train = train.with_columns(pl.col("category").cast(dt))
        val = val.with_columns(pl.col("category").cast(dt))

    y_tr = rng.integers(0, 2, args.n_train_slice).astype(np.int8)
    y_v = rng.integers(0, 2, args.n_val_slice).astype(np.int8)

    print(f"  train shape={train.shape}, val shape={val.shape}", flush=True)
    print("  Calling XGB.fit() — expect silent kill in 'categorical' mode on "
          "Windows+xgboost 3.2.0 if multi-column cast + slice is the trigger.",
          flush=True)
    t0 = time.perf_counter()
    m = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1,
        max_cat_to_onehot=1, max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    try:
        m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
        print(f"  FIT_OK in {time.perf_counter()-t0:.1f}s (bug did NOT reproduce)",
              flush=True)
    except BaseException as e:
        print(f"  RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
