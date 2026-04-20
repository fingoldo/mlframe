"""Reproducer for XGBoost 3.2 behavior when val DataFrame contains
Polars Categorical values absent from train's per-Series dict.

==============================================================================
TL;DR — observed behaviors by scale
==============================================================================

At small-to-medium scale (500k — 2M rows × 5-19 cat features on
Windows + 17 GB RAM), XGB raises a **clean Python exception**:

    xgboost.core.XGBoostError: [...]/cat_container.h:29:
    Found a category not in the training set for the 0th (0-based)
    column: `cat_a_v0010`

This is correct, documented, catchable behavior. The guard at
``data/cat_container.h:29`` is working.

At **production scale** (7.3M train × 812k val × 19 cat features on
Windows + 128 GB RAM) the same setup **silently kills the process**:

  - Train IterativeDMatrix is logged as built.
  - Val IterativeDMatrix is never logged.
  - No Python exception surfaces.
  - With ``SetErrorMode(SEM_NOGPFAULTERRORBOX)``: kernel exits 1.
  - Without it: Windows WER modal "Python has stopped working".

So the ``cat_container.h:29`` guard either isn't firing at scale, or
is firing in a worker thread whose exception propagation fails,
converting to ``std::abort()`` / SIGABRT.

Workaround (validated at every scale tested, including prod): cast
all splits to a shared ``pl.Enum(sorted(union_of_all_values))``.
Physical codes become consistent across Series, the guard doesn't
need to fire at all, XGB proceeds normally. MakeCuts time in prod
logs drops **50×** after alignment (0.9s → 18ms, matching the
no-categorical baseline), suggesting XGB takes a different (faster)
internal path when dicts are pre-aligned.

==============================================================================
Environment where prod silent-kill was observed
==============================================================================

  Windows 10 Pro 10.0.19045
  Python 3.11.5
  polars 1.35.2
  xgboost 3.2.0
  128 GB RAM, 237 GB pagefile

Environment where clean XGBoostError was observed (small/medium):

  Windows 10, Python 3.11.5, polars 1.35.2, xgboost 3.2.0
  17 GB RAM, 11 GB pagefile

==============================================================================
Usage
==============================================================================

    python xgb_polars_categorical_scale_crash.py [--scale prod|small]
                                                 [--mode crash|workaround]

    --scale prod       → 7.3M train + 812k val, 19 cat features (matches prod)
    --scale small      → 500k train + 56k val, 5 cat features (may not crash)
    --mode crash       → no alignment, expected to crash at prod scale
    --mode workaround  → pl.Enum(union) alignment, expected to pass

Default: ``--scale prod --mode crash``. Add ``--verbosity 3`` for
XGB C++ debug output. Stdout is unbuffered so the last printed line
before process death is visible.

==============================================================================
What to look for
==============================================================================

Crash mode (``--mode crash``) expected output:

    [prod] Built train frame 7300000x20 in Ns
    [prod] Built val frame   812000x20 in Ns
    [prod] train categorical columns:
        _raw_tags: pl.Categorical, train n_unique=277, val n_unique=314,
                   shared=277, val-only=37, train-only=0
        occupation: ...
    [prod] Calling XGB fit...
    [XX:XX:XX] ======== Monitor (0): HostSketchContainer ========
    [XX:XX:XX] AllReduce: 0.02s, 1 calls
    [XX:XX:XX] MakeCuts: 0.9s, 1 calls                              ← note: slow
    [XX:XX:XX] INFO: iterative_dmatrix.cc: Finished constructing ...
                                            the `IterativeDMatrix`:
                                            (7300000, 19, ...).     ← train OK
    <<< process dies, no further output, exit code != 0 >>>

Workaround mode expected output:

    [prod] Aligning categorical dicts via pl.Enum(union) ...
    [prod] Calling XGB fit...
    [XX:XX:XX] MakeCuts: 0.02s, 1 calls                             ← ~50x faster
    [XX:XX:XX] Finished constructing ... (7300000, 19, ...).         ← train OK
    [XX:XX:XX] Finished constructing ... (812000, 19, ...).          ← val OK
    [XX:XX:XX] ======== Monitor (0): GBTree ========
    ... normal training proceeds ...
    Fit completed in Xs, best_iter=N

==============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import polars as pl
from xgboost import XGBClassifier

# Stdout unbuffered so last-before-crash lines actually reach the log.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

try:
    import faulthandler
    faulthandler.enable()  # SIGSEGV/SIGABRT -> Python traceback to stderr
except Exception:
    pass

# Windows: suppress "Program has stopped working" WER popup so kernel exits
# with a non-zero code instead of hanging on a modal dialog.
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


# -----------------------------------------------------------------------------
# Cardinalities matching a real production prod_jobsdetails dataset.
# Each entry = (feature_name, n_unique_train, n_unique_val_extra).
# "train" produces n_unique_train distinct strings.
# "val" picks from the same pool PLUS n_unique_val_extra fresh strings
# that train never sees — models time-ordered splits where new category
# codes appear in the later period.
# -----------------------------------------------------------------------------
PROD_CARDINALITIES: List[Tuple[str, int, int]] = [
    ("_raw_tags",           277,   37),
    ("_raw_languages",     1769,  147),
    ("occupation",          222,    0),
    ("job_post_source",      95,    2),
    ("category",             89,    0),
    ("job_post_browser",     19,    0),
    ("job_post_device",      13,    1),
    ("category_group",       13,    0),
    ("job_local_flexibility", 6,   0),
    ("job_post_type",         5,   0),
    ("job_req_english",       5,   0),
    ("job_urgency",           5,   0),
    ("workload",              5,   0),
    ("hourly_budget_type",    4,   0),
    ("contractor_tier",       4,   0),
    ("desc_ai_opted_in",      3,   0),
    ("job_post_flow_type",    3,   0),
    ("qual_type",             3,   0),
    ("job_type",              2,   0),
]

SMALL_CARDINALITIES: List[Tuple[str, int, int]] = [
    ("cat_a", 100, 15),
    ("cat_b",  50,  5),
    ("cat_c",  20,  0),
    ("cat_d",  10,  2),
    ("cat_e",   5,  0),
]


def build_split(
    n_rows: int,
    cardinalities: List[Tuple[str, int, int]],
    *,
    include_val_extras: bool,
    seed: int,
) -> pl.DataFrame:
    """Build a DataFrame with the specified categorical columns.

    If ``include_val_extras=True``, the per-column value pool includes
    the train-only + val-extra strings.  If False, only the train pool.
    Nulls are NOT injected (keeps the repro focused on the dict-mismatch
    mechanism, not round-11's null-in-Categorical bug).
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for name, n_train, n_val_extra in cardinalities:
        train_pool = [f"{name}_t{i:04d}" for i in range(n_train)]
        val_extras = [f"{name}_v{i:04d}" for i in range(n_val_extra)]
        pool = train_pool + val_extras if include_val_extras else train_pool
        # Shuffle so physical codes (order-of-first-occurrence) are RANDOMIZED
        # per Series. This is the documented polars.Categorical behaviour and
        # the direct trigger: train and val end up with different physical
        # codes for the same strings.
        vals = rng.choice(pool, size=n_rows).tolist()
        cols[name] = pl.Series(name, vals, dtype=pl.Categorical)
    # One numeric feature so the model has something to split on.
    cols["num_f"] = rng.standard_normal(n_rows).astype(np.float32)
    return pl.DataFrame(cols)


def align_categorical_dicts(
    train: pl.DataFrame, val: pl.DataFrame, cat_cols: List[str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Workaround: cast each cat column to ``pl.Enum(sorted union)``.

    Shared Enum dict → consistent physical codes across both frames.
    """
    for col in cat_cols:
        tr_u = set(train.select(pl.col(col).drop_nulls().unique())[col].to_list())
        v_u  = set(val.select(pl.col(col).drop_nulls().unique())[col].to_list())
        union = sorted(tr_u | v_u)
        enum_dt = pl.Enum(union)
        train = train.with_columns(pl.col(col).cast(enum_dt))
        val   = val.with_columns(pl.col(col).cast(enum_dt))
    return train, val


def report_dict_mismatch(
    train: pl.DataFrame, val: pl.DataFrame, cat_cols: List[str],
) -> None:
    print("[repro] per-column dict comparison:")
    for col in cat_cols:
        tr_u = set(train.select(pl.col(col).unique())[col].to_list())
        v_u  = set(val.select(pl.col(col).unique())[col].to_list())
        shared = tr_u & v_u
        val_only = v_u - tr_u
        train_only = tr_u - v_u
        print(
            f"    {col}: train n_unique={len(tr_u)}, val n_unique={len(v_u)}, "
            f"shared={len(shared)}, val-only={len(val_only)}, "
            f"train-only={len(train_only)}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=("prod", "small"), default="prod",
                    help="prod: 7.3M/812k/19 cats (crashes). small: 500k/56k/5 cats.")
    ap.add_argument("--n-train", type=int, default=None,
                    help="Override train row count (useful for scale escalation).")
    ap.add_argument("--n-val", type=int, default=None,
                    help="Override val row count.")
    ap.add_argument("--mode", choices=("crash", "workaround"), default="crash",
                    help="crash: no alignment. workaround: pl.Enum(union) alignment.")
    ap.add_argument("--verbosity", type=int, default=2,
                    help="XGB verbosity 0-3 (3=debug with C++ internals).")
    ap.add_argument("--n-estimators", type=int, default=20,
                    help="XGB n_estimators (low for fast repro).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.scale == "prod":
        n_train, n_val = 7_300_000, 812_000
        cardinalities = PROD_CARDINALITIES
    else:
        n_train, n_val = 500_000, 56_000
        cardinalities = SMALL_CARDINALITIES
    # Allow CLI override for scale escalation without changing preset.
    if args.n_train is not None:
        n_train = args.n_train
    if args.n_val is not None:
        n_val = args.n_val

    import xgboost, polars
    print(f"[repro] env: python {sys.version.split()[0]}, "
          f"polars {polars.__version__}, xgboost {xgboost.__version__}, "
          f"platform {sys.platform}")
    print(f"[repro] scale={args.scale}, mode={args.mode}, "
          f"n_train={n_train:_}, n_val={n_val:_}, "
          f"n_cats={len(cardinalities)}")

    t0 = time.perf_counter()
    # Train: only train-pool strings.
    train = build_split(n_train, cardinalities,
                        include_val_extras=False, seed=args.seed)
    print(f"[repro] Built train {train.shape} in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    # Val: train-pool + val-extras.
    val = build_split(n_val, cardinalities,
                      include_val_extras=True, seed=args.seed + 1)
    print(f"[repro] Built val   {val.shape} in {time.perf_counter()-t0:.1f}s")

    cat_cols = [c for c, _, _ in cardinalities]
    report_dict_mismatch(train, val, cat_cols)

    if args.mode == "workaround":
        t0 = time.perf_counter()
        train, val = align_categorical_dicts(train, val, cat_cols)
        print(f"[repro] Aligned categorical dicts via pl.Enum(union) "
              f"in {time.perf_counter()-t0:.1f}s")
        print(f"[repro] After alignment, example dtype: "
              f"{train.schema[cat_cols[0]]}")

    # Labels — simple deterministic ~50/50 binary target.
    rng = np.random.default_rng(args.seed + 2)
    y_train = rng.integers(0, 2, size=n_train).astype(np.int8)
    y_val   = rng.integers(0, 2, size=n_val).astype(np.int8)

    os.environ.setdefault("XGBOOST_VERBOSITY", str(args.verbosity))

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.1,
        enable_categorical=True,
        max_cat_to_onehot=1,
        max_cat_threshold=100,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        early_stopping_rounds=max(2, args.n_estimators // 3),
        random_state=args.seed,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=args.verbosity,
    )

    print(f"[repro] Calling XGB fit — if this is mode=crash + scale=prod on "
          f"Windows, expect process death after train IterativeDMatrix is "
          f"built but before val IterativeDMatrix is built.")
    t0 = time.perf_counter()
    model.fit(train, y_train, eval_set=[(val, y_val)], verbose=False)
    elapsed = time.perf_counter() - t0
    print(f"[repro] Fit completed in {elapsed:.1f}s, "
          f"best_iter={model.best_iteration}")


if __name__ == "__main__":
    main()
