r"""Pure synthetic reproducer for XGBoost 3.2 access violation
(Windows 0xC0000005) on pl.Categorical columns with sparse physical
codes.

Background
----------
Bisection on a real production workload (mlframe + prod_jobsdetails,
7.3M rows × ~100 cols, Windows + 128 GB RAM) isolated the trigger
down to a single pl.Categorical column. The visible state right
before XGB.fit:

    train[category]: n_chunks=1, n_unique=49,
        physical_codes_range=[2, 3287945], distinct_codes=49

89 unique string values in the source, but physical codes inherited
from a polars-global StringCache that had ~3.3M distinct entries
from OTHER columns previously cast to Categorical in the same
pipeline. So 49 codes scattered across the [0, 3287945] range.

This script reproduces the pathology WITHOUT needing the original
dataset: we pollute the cache with padding strings first, then build
the crashing column, then slice off the padding.

Usage
-----
    python -m mlframe.profiling.repro_xgb_sparse_categorical_codes \
        [--padding-uniques N]   # default 3_000_000
        [--n-used N]            # default 49
        [--n-train N]           # default 211_168
        [--mode sparse|enum]    # default sparse (crashes)

Expected on Windows + xgboost 3.2.0 + polars 1.35:
    mode=sparse  → silent kill, exit 3221226505 (0xC0000005)
    mode=enum    → fit completes normally

If ``--padding-uniques`` is too small, sparse codes aren't sparse
enough to trigger; try values in the 1M-5M range.
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


def build_sparse_categorical(
    n_rows: int, n_used: int, padding_uniques: int, seed: int,
) -> pl.Series:
    """Build a pl.Categorical with SCATTERED physical codes (not just
    shifted — actually spread across the 0..padding_uniques range).

    Strategy: preamble alternates groups of padding strings with
    individual used-value strings. After Categorical cast, used
    values get physical codes at positions ``K, 2K+1, 3K+2, ...``
    where K = padding_uniques / n_used. The resulting codes span
    the full range [K, padding_uniques] with ~K gaps between them —
    matching the user's real production pattern of codes in
    [2, 3_287_945] with only 49 distinct.
    """
    rng = np.random.default_rng(seed)
    used = [f"c_{i:03d}" for i in range(n_used)]
    per_slot = max(1, padding_uniques // (n_used + 1))

    preamble: list[str] = []
    pad_idx = 0
    for i in range(n_used):
        for _ in range(per_slot):
            preamble.append(f"__pad_{pad_idx}")
            pad_idx += 1
        preamble.append(used[i])
    while pad_idx < padding_uniques:
        preamble.append(f"__pad_{pad_idx}")
        pad_idx += 1

    data = rng.choice(used, size=n_rows).tolist()
    full = preamble + data
    ser = pl.Series("category", full, dtype=pl.Categorical)
    # Slice off preamble — dict is preserved; the returned Series
    # references only ``used`` strings but their physical codes are
    # scattered because they were assigned during the preamble scan.
    return ser[len(preamble):]


def build_compact_categorical(
    n_rows: int, n_used: int, seed: int,
) -> pl.Series:
    rng = np.random.default_rng(seed)
    used = [f"c_{i:03d}" for i in range(n_used)]
    return pl.Series("category", rng.choice(used, size=n_rows).tolist(),
                     dtype=pl.Categorical)


def run(
    n_train: int, n_val: int, n_used: int, padding_uniques: int,
    n_numeric: int, mode: str, seed: int,
):
    import xgboost, polars
    print(f"env: python {sys.version.split()[0]}, polars {polars.__version__}, "
          f"xgboost {xgboost.__version__}, platform {sys.platform}")
    print(f"=== n_train={n_train:_}, n_val={n_val:_}, n_used={n_used}, "
          f"padding_uniques={padding_uniques:_}, n_numeric={n_numeric}, "
          f"mode={mode} ===", flush=True)

    t0 = time.perf_counter()
    if mode == "sparse":
        cat_tr = build_sparse_categorical(n_train, n_used, padding_uniques, seed)
        cat_va = build_sparse_categorical(n_val,   n_used, padding_uniques, seed + 1)
    elif mode == "enum":
        # Control: explicit Enum with compact codes by construction.
        tokens = [f"c_{i:03d}" for i in range(n_used)]
        rng = np.random.default_rng(seed)
        cat_tr = pl.Series("category", rng.choice(tokens, size=n_train).tolist(),
                           dtype=pl.Enum(tokens))
        rng2 = np.random.default_rng(seed + 1)
        cat_va = pl.Series("category", rng2.choice(tokens, size=n_val).tolist(),
                           dtype=pl.Enum(tokens))
    else:
        raise SystemExit(f"unknown mode: {mode}")
    print(f"  built cats in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  train[category]: dtype={cat_tr.dtype}, n_unique={cat_tr.n_unique()}, "
          f"physical_codes_range=[{cat_tr.to_physical().min()}, "
          f"{cat_tr.to_physical().max()}], "
          f"distinct_codes={cat_tr.to_physical().n_unique()}", flush=True)

    rng = np.random.default_rng(seed + 2)
    tr = pl.DataFrame({"category": cat_tr})
    va = pl.DataFrame({"category": cat_va})
    for i in range(n_numeric):
        tr = tr.with_columns(pl.Series(f"num_{i}",
            rng.standard_normal(n_train).astype(np.float32)))
        va = va.with_columns(pl.Series(f"num_{i}",
            rng.standard_normal(n_val).astype(np.float32)))
    y_tr = rng.integers(0, 2, n_train).astype(np.int8)
    y_va = rng.integers(0, 2, n_val).astype(np.int8)

    print(f"  train shape={tr.shape}, val shape={va.shape}", flush=True)
    print("  Calling XGBClassifier.fit() — sparse mode expected to silently "
          "kill (exit 3221226505) on Windows + xgboost 3.2.0.", flush=True)
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
    try:
        model.fit(tr, y_tr, eval_set=[(va, y_va)], verbose=False)
        print(f"  FIT_OK in {time.perf_counter()-t0:.1f}s, "
              f"best_iter={model.best_iteration}  (bug did NOT reproduce)",
              flush=True)
    except BaseException as e:
        print(f"  RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sparse", "enum"), default="sparse")
    ap.add_argument("--padding-uniques", type=int, default=3_000_000)
    ap.add_argument("--n-used", type=int, default=49)
    ap.add_argument("--n-train", type=int, default=211_168)
    ap.add_argument("--n-val", type=int, default=100_000)
    ap.add_argument("--n-numeric", type=int, default=95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.n_train, args.n_val, args.n_used, args.padding_uniques,
        args.n_numeric, args.mode, args.seed)
