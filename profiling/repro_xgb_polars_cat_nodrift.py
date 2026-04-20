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


def build_frame(n_rows: int, seed: int, dtype_mode: str) -> pl.DataFrame:
    """Build a frame with the shared pool per column. ``dtype_mode``
    selects the Polars categorical representation:
      - ``categorical`` = ``pl.Categorical`` (per-Series auto dict)
      - ``enum``        = ``pl.Enum(pool)`` (shared dict, deterministic)
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for name, k in CARDINALITIES:
        pool = [f"{name}_v{i:04d}" for i in range(k)]
        vals = rng.choice(pool, size=n_rows).tolist()
        if dtype_mode == "categorical":
            cols[name] = pl.Series(name, vals, dtype=pl.Categorical)
        elif dtype_mode == "enum":
            cols[name] = pl.Series(name, vals, dtype=pl.Enum(pool))
        else:
            raise ValueError(f"dtype_mode must be 'categorical' or 'enum', got {dtype_mode!r}")
    cols["num_f"] = rng.standard_normal(n_rows).astype(np.float32)
    return pl.DataFrame(cols)


def run(n_train: int, n_val: int, dtype_mode: str) -> None:
    import xgboost, polars
    print(f"env: python {sys.version.split()[0]}, polars {polars.__version__}, "
          f"xgboost {xgboost.__version__}, platform {sys.platform}")
    print(f"\n=== n_train={n_train:_}, n_val={n_val:_}, dtype_mode={dtype_mode} ===")

    t0 = time.perf_counter()
    tr = build_frame(n_train, seed=42, dtype_mode=dtype_mode)
    va = build_frame(n_val, seed=43, dtype_mode=dtype_mode)
    print(f"  built frames in {time.perf_counter()-t0:.1f}s")
    print(f"  train sample dtype ({CARDINALITIES[0][0]}): {tr[CARDINALITIES[0][0]].dtype}")

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
    mode = sys.argv[1] if len(sys.argv) > 1 else "categorical"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    run(n_train, n_val, mode)
