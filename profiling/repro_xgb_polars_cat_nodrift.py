"""Probe: does XGB 3.x silently kill on large pl.Categorical frames
*without* any train/val category drift?

Background
----------
On prod_jobsdetails 2026-04-20 (7.3M train x 812k val x 19 cat features,
Windows, 128 GB RAM), XGB with ``enable_categorical=True`` silently
killed the Jupyter kernel between train IterativeDMatrix construction
and val IterativeDMatrix construction.

First hypothesis (round 18): train/val dict mismatch on ``pl.Categorical``
causes XGB to read val's physical codes as indices into train's bin
structure and corrupt memory. Casting all splits to a shared
``pl.Enum(union)`` was an empirically-working workaround.

But a follow-up prod run (2026-04-20 09:01) with **drift columns removed**
(``Category drift: (none)`` in the log) and ``align_polars_categorical_dicts=False``
**still silently killed the kernel**. So unseen categories are NOT the
primary trigger, and round 18's Enum cast is fixing something else —
maybe merely that casting to ``pl.Enum`` produces a different Arrow
memory layout than ``pl.Categorical``, one XGB tolerates at scale.

Local repro on a 17 GB / 4-core Windows machine up to 5M rows / 19 cats
with no drift passed cleanly for both dtype modes. This script is
intended to be run on the same 128 GB machine where the silent kill was
observed, to answer: is it ``pl.Categorical`` at prod scale that breaks,
or is something more specific (drift + scale? specific data layout?).

What to run on the prod box
---------------------------

    python -m mlframe.profiling.repro_xgb_polars_cat_nodrift categorical 7000000 800000
    python -m mlframe.profiling.repro_xgb_polars_cat_nodrift enum        7000000 800000

If the first silently kills and the second passes, the bug is
``pl.Categorical``-specific regardless of drift, and we have a clean
reproducer for a bug report to xgboost (or to polars — the diagnosis
isn't yet conclusive).

If the first passes, the prod trigger requires something this script
doesn't reproduce (drift + pl.Categorical + scale, a specific row
pattern, OMP thread count interaction, etc.).
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import polars as pl
from xgboost import XGBClassifier

# Fault handling: on Windows, SetErrorMode suppresses the
# "Python has stopped working" modal so the kernel exits non-zero
# instead of hanging. faulthandler dumps a Python traceback on any
# fatal signal it can catch (SEGV/FPE/ABRT/BUS/ILL — including from
# OMP worker threads when all_threads=True).
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass
if sys.platform == "win32":
    try:
        import ctypes
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
        )
    except Exception:
        pass

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Cardinalities matching prod_jobsdetails, shared between train and val
# (no drift — both splits sample from the same value pool per column).
CARDINALITIES = [
    ("_raw_tags",            277),
    ("_raw_languages",      1769),
    ("occupation",           222),
    ("job_post_source",       95),
    ("category",              89),
    ("job_post_browser",      19),
    ("job_post_device",       13),
    ("category_group",        13),
    ("job_local_flexibility",  6),
    ("job_post_type",          5),
    ("job_req_english",        5),
    ("job_urgency",            5),
    ("workload",               5),
    ("hourly_budget_type",     4),
    ("contractor_tier",        4),
    ("desc_ai_opted_in",       3),
    ("job_post_flow_type",     3),
    ("qual_type",              3),
    ("job_type",               2),
]


def build_frame(
    n_rows: int, seed: int, dtype_mode: str,
    n_numeric: int = 1, fill_null_rate: float = 0.0,
) -> pl.DataFrame:
    """Build a frame with the shared pool per column.

    ``dtype_mode``     : ``categorical`` (pl.Categorical, per-Series auto
                         dict) or ``enum`` (pl.Enum(pool), shared dict).
    ``n_numeric``      : number of extra Float32 columns to add. Prod uses
                         ~95 numerics; the original probe used 1. Bumping
                         this matters because XGB histogram buffers scale
                         with n_features.
    ``fill_null_rate`` : if > 0, inject nulls into each categorical at
                         this rate, then ``fill_null('__MISSING__')``.
                         Polars auto-extends the Arrow dict with the
                         sentinel, producing a potentially-pathological
                         dict state that mlframe's round-17 fill does
                         in production (but the original probe skipped).
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for name, k in CARDINALITIES:
        pool = [f"{name}_v{i:04d}" for i in range(k)]
        vals = rng.choice(pool, size=n_rows).tolist()
        # Optionally inject nulls to exercise the fill_null path.
        if fill_null_rate > 0 and dtype_mode == "categorical":
            # Cannot inject None into pl.Enum (must be a declared category);
            # for Categorical this is fine, Polars represents it via the
            # validity bitmap.
            null_mask = rng.random(n_rows) < fill_null_rate
            vals = [None if m else v for v, m in zip(vals, null_mask)]
        if dtype_mode == "categorical":
            s = pl.Series(name, vals, dtype=pl.Categorical)
            if fill_null_rate > 0:
                # This mirrors mlframe's round-17 fill: extends the dict
                # with the sentinel, drops the validity bitmap.
                s = s.fill_null("__MISSING__")
            cols[name] = s
        elif dtype_mode == "enum":
            cols[name] = pl.Series(name, vals, dtype=pl.Enum(pool))
        else:
            raise ValueError(f"dtype_mode must be 'categorical' or 'enum', got {dtype_mode!r}")
    # Extra numeric columns — default 1 matches original probe for
    # backward-compat. Pass --n-numeric 95 for prod-scale width.
    for i in range(n_numeric):
        cols[f"num_f{i}"] = rng.standard_normal(n_rows).astype(np.float32)
    return pl.DataFrame(cols)


def build_frame_prod_dtypes(
    n_rows: int, seed: int, dtype_mode: str,
    fill_null_rate: float = 0.0,
) -> pl.DataFrame:
    """Prod-matched dtype mix: 15 Boolean, 38 Float32, 35 Int16, 2 Int32,
    2 Int64, 2 UInt32, plus the 19 Categoricals from CARDINALITIES.

    This variant exists because the basic ``build_frame(..., n_numeric=95)``
    didn't reproduce the silent kill at 7M on the prod box. Next
    hypothesis: the trigger is dtype-mix-specific — XGB's hist path
    with a non-float-only dtype mix allocates / iterates differently
    and hits the bug only when multiple numpy-width dtypes coexist.
    """
    rng = np.random.default_rng(seed)
    cols = {}
    # Categoricals (shared pool — no drift).
    for name, k in CARDINALITIES:
        pool = [f"{name}_v{i:04d}" for i in range(k)]
        vals = rng.choice(pool, size=n_rows).tolist()
        if fill_null_rate > 0 and dtype_mode == "categorical":
            null_mask = rng.random(n_rows) < fill_null_rate
            vals = [None if m else v for v, m in zip(vals, null_mask)]
        if dtype_mode == "categorical":
            s = pl.Series(name, vals, dtype=pl.Categorical)
            if fill_null_rate > 0:
                s = s.fill_null("__MISSING__")
            cols[name] = s
        elif dtype_mode == "enum":
            cols[name] = pl.Series(name, vals, dtype=pl.Enum(pool))
        else:
            raise ValueError(dtype_mode)
    # 15 Boolean
    for i in range(15):
        cols[f"bool_{i}"] = (rng.random(n_rows) > 0.5).astype(bool)
    # 38 Float32
    for i in range(38):
        cols[f"num_f32_{i}"] = rng.standard_normal(n_rows).astype(np.float32)
    # 35 Int16
    for i in range(35):
        cols[f"num_i16_{i}"] = rng.integers(-1000, 1000, size=n_rows).astype(np.int16)
    # 2 Int32
    for i in range(2):
        cols[f"num_i32_{i}"] = rng.integers(0, 1_000_000, size=n_rows).astype(np.int32)
    # 2 Int64
    for i in range(2):
        cols[f"num_i64_{i}"] = rng.integers(0, 10_000_000, size=n_rows).astype(np.int64)
    # 2 UInt32
    for i in range(2):
        cols[f"num_u32_{i}"] = rng.integers(0, 1_000_000, size=n_rows).astype(np.uint32)
    return pl.DataFrame(cols)


def run(n_train: int, n_val: int, dtype_mode: str,
        n_numeric: int = 1, fill_null_rate: float = 0.0,
        prod_dtypes: bool = False) -> None:
    import xgboost, polars
    print(f"env: python {sys.version.split()[0]}, polars {polars.__version__}, "
          f"xgboost {xgboost.__version__}, platform {sys.platform}")
    print(f"\n=== n_train={n_train:_}, n_val={n_val:_}, dtype_mode={dtype_mode}, "
          f"n_numeric={n_numeric}, fill_null_rate={fill_null_rate}, "
          f"prod_dtypes={prod_dtypes} ===")

    t0 = time.perf_counter()
    if prod_dtypes:
        tr = build_frame_prod_dtypes(n_train, seed=42, dtype_mode=dtype_mode,
                                     fill_null_rate=fill_null_rate)
        va = build_frame_prod_dtypes(n_val, seed=43, dtype_mode=dtype_mode,
                                     fill_null_rate=fill_null_rate)
    else:
        tr = build_frame(n_train, seed=42, dtype_mode=dtype_mode,
                         n_numeric=n_numeric, fill_null_rate=fill_null_rate)
        va = build_frame(n_val, seed=43, dtype_mode=dtype_mode,
                         n_numeric=n_numeric, fill_null_rate=fill_null_rate)
    print(f"  built frames in {time.perf_counter()-t0:.1f}s, shape={tr.shape}")
    # Count dtypes to confirm mix.
    dt_counts = {}
    for dt in tr.dtypes:
        key = str(dt).split("(")[0]
        dt_counts[key] = dt_counts.get(key, 0) + 1
    print(f"  dtype counts: {dt_counts}")

    # Sanity — confirm no drift exists between splits (same pool used).
    for name, _ in CARDINALITIES[:3]:
        tr_u = set(tr[name].unique().to_list())
        va_u = set(va[name].unique().to_list())
        print(f"  {name}: train card={len(tr_u)}, val card={len(va_u)}, "
              f"val_only={len(va_u - tr_u)}")

    y_tr = np.random.default_rng(44).integers(0, 2, size=n_train).astype(np.int8)
    y_va = np.random.default_rng(45).integers(0, 2, size=n_val).astype(np.int8)

    os.environ.setdefault("XGBOOST_VERBOSITY", "1")
    model = XGBClassifier(
        n_estimators=5,
        learning_rate=0.1,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        early_stopping_rounds=3,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=1,
    )
    print("  calling model.fit(...)")
    t0 = time.perf_counter()
    try:
        model.fit(tr, y_tr, eval_set=[(va, y_va)], verbose=False)
        print(f"  [OK] fit completed in {time.perf_counter()-t0:.1f}s, "
              f"best_iter={model.best_iteration}")
    except BaseException as e:
        print(f"  [RAISED] {type(e).__name__}: {e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="categorical",
                    choices=("categorical", "enum"))
    ap.add_argument("n_train", nargs="?", type=int, default=2_000_000)
    ap.add_argument("n_val",   nargs="?", type=int, default=200_000)
    ap.add_argument("--n-numeric", type=int, default=1,
                    help="Extra Float32 columns. Use 95 to match prod_jobsdetails width.")
    ap.add_argument("--fill-null-rate", type=float, default=0.0,
                    help="If >0, inject nulls at this rate into each Categorical "
                         "then fill_null('__MISSING__'). Matches mlframe's round-17 fill.")
    ap.add_argument("--prod-dtypes", action="store_true",
                    help="Use prod dtype mix (15 Bool, 38 Float32, 35 Int16, 2 Int32, "
                         "2 Int64, 2 UInt32, plus 19 Categorical). Overrides --n-numeric.")
    args = ap.parse_args()
    run(args.n_train, args.n_val, args.mode,
        n_numeric=args.n_numeric, fill_null_rate=args.fill_null_rate,
        prod_dtypes=args.prod_dtypes)
