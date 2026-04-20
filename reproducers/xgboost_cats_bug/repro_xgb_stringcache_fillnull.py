r"""Pure synthetic reproducer v4: sparse Categorical physical codes via
pl.enable_string_cache() + fill_null.

Trace on prod data (2026-04-20) identified the exact trigger:

  Step 8  [category dtype=Categorical, codes=[103, 455]]
  Step 9  fill_null('__MISSING__')
  Step 9  [category dtype=Categorical, codes=[103, 3287945]]   <-- JUMP

With ``pl.using_string_cache() == True`` (enabled globally somewhere
in the user's environment — not their visible code, probably mlframe
or an imported module does it), ``fill_null`` on a Categorical
re-resolves the column's values through the *global* StringCache.
That cache has ~3.3M entries from all the other string columns cast
earlier, so the 89 category values get scattered physical codes in
[103, 3_287_945] instead of the compact [103, 455] they had
immediately after parquet load.

XGB's enable_categorical=True with tree_method=hist indexes bin
storage by physical code. With codes up to 3.3M but only ~49 bins
allocated, the index is out-of-bounds → memory corruption →
Windows SEH access violation (0xC0000005).

Usage:
    python -m mlframe.profiling.repro_xgb_stringcache_fillnull

Expected on Windows + xgboost 3.2.0 + polars 1.35:
    exit 3221226505 (0xC0000005 access violation), no traceback.

Workaround: don't enable the global StringCache. Or if already
enabled, cast to pl.Enum(sorted_values) before fit so the dict is
compact by construction.
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


def run(n_train: int, n_val: int, cache_pollution: int, n_used: int,
        n_numeric: int, fix_enum: bool, seed: int):
    import xgboost
    print(f"env: python {sys.version.split()[0]}, polars {pl.__version__}, "
          f"xgboost {xgboost.__version__}, platform {sys.platform}", flush=True)

    # CRITICAL: activate global string cache — this is the hidden
    # prerequisite. On prod the user's environment had
    # pl.using_string_cache() == True but not from their visible code.
    pl.enable_string_cache()
    print(f"pl.using_string_cache() = {pl.using_string_cache()}", flush=True)

    rng = np.random.default_rng(seed)

    # Step 1: pollute the StringCache by casting a big Series that
    # INTERLEAVES padding strings with the real-used strings. This
    # produces scattered physical codes for real_NNN — matching the
    # prod observation of codes in [103, 3_287_945] with 89 distinct,
    # not just a contiguous shifted range.
    used = [f"real_{i:03d}" for i in range(n_used)]
    print(f"Polluting StringCache with {cache_pollution:_} padding + "
          f"interleaved real strings...", flush=True)
    t0 = time.perf_counter()
    mixed = [f"pad_{i:08d}" for i in range(cache_pollution)] + used
    rng.shuffle(mixed)
    _ = pl.Series("_pollute", mixed, dtype=pl.Categorical)
    # Confirm scatter: check real_000..real_N's physical codes after pollution.
    check = pl.Series("_check", used, dtype=pl.Categorical)
    ch_codes = check.to_physical().to_list()
    print(f"  done in {time.perf_counter()-t0:.1f}s. "
          f"real_NNN codes in global cache: min={min(ch_codes)}, "
          f"max={max(ch_codes)}, distinct={len(set(ch_codes))}", flush=True)
    print(f"Building train/val with {n_used} real categories...", flush=True)
    data_tr = rng.choice(used, size=n_train).tolist()
    data_v  = rng.choice(used, size=n_val).tolist()
    # Inject a few nulls so fill_null has something to replace.
    null_mask_tr = rng.random(n_train) < 0.01
    null_mask_v  = rng.random(n_val)   < 0.01
    data_tr = [None if m else v for v, m in zip(data_tr, null_mask_tr)]
    data_v  = [None if m else v for v, m in zip(data_v,  null_mask_v)]

    cat_tr = pl.Series("category", data_tr, dtype=pl.Categorical)
    cat_v  = pl.Series("category", data_v,  dtype=pl.Categorical)
    print(f"  after cast: train[category] codes=["
          f"{cat_tr.to_physical().min()}, {cat_tr.to_physical().max()}], "
          f"distinct={cat_tr.to_physical().n_unique()}", flush=True)

    # Step 3: fill_null (the actual re-materialization trigger).
    cat_tr = cat_tr.fill_null("__MISSING__")
    cat_v  = cat_v.fill_null("__MISSING__")
    print(f"  after fill_null: train[category] codes=["
          f"{cat_tr.to_physical().min()}, {cat_tr.to_physical().max()}], "
          f"distinct={cat_tr.to_physical().n_unique()}", flush=True)

    if fix_enum:
        print("  APPLYING WORKAROUND: cast to pl.Enum(union)...", flush=True)
        tr_u = set(cat_tr.drop_nulls().unique().to_list())
        v_u  = set(cat_v.drop_nulls().unique().to_list())
        union = sorted(tr_u | v_u)
        dt = pl.Enum(union)
        cat_tr = cat_tr.cast(dt)
        cat_v  = cat_v.cast(dt)
        print(f"  after enum: train[category] codes=["
              f"{cat_tr.to_physical().min()}, {cat_tr.to_physical().max()}], "
              f"distinct={cat_tr.to_physical().n_unique()}", flush=True)

    # Build frames.
    tr = pl.DataFrame({"category": cat_tr})
    va = pl.DataFrame({"category": cat_v})
    for i in range(n_numeric):
        tr = tr.with_columns(pl.Series(f"num_{i}",
            rng.standard_normal(n_train).astype(np.float32)))
        va = va.with_columns(pl.Series(f"num_{i}",
            rng.standard_normal(n_val).astype(np.float32)))
    y_tr = rng.integers(0, 2, n_train).astype(np.int8)
    y_v  = rng.integers(0, 2, n_val).astype(np.int8)

    print(f"\ntrain shape={tr.shape}, val shape={va.shape}", flush=True)
    print("Calling XGBClassifier.fit() — expect silent kill in 'sparse' mode "
          "on Windows+xgboost 3.2.0.", flush=True)
    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1,
        max_cat_to_onehot=1, max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    try:
        model.fit(tr, y_tr, eval_set=[(va, y_v)], verbose=False)
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s "
              f"(bug did NOT reproduce — unexpected)", flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=211_168)
    ap.add_argument("--n-val", type=int, default=100_000)
    ap.add_argument("--cache-pollution", type=int, default=3_000_000,
                    help="Number of unique strings to inject into the global "
                         "StringCache before building the crashing column.")
    ap.add_argument("--n-used", type=int, default=89)
    ap.add_argument("--n-numeric", type=int, default=95)
    ap.add_argument("--fix-enum", action="store_true",
                    help="Apply pl.Enum(union) workaround after fill_null; "
                         "expect FIT_OK.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.n_train, args.n_val, args.cache_pollution, args.n_used,
        args.n_numeric, args.fix_enum, args.seed)
