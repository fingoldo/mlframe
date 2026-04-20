r"""Bisect which columns AND rows of ``jobs_details.parquet`` trigger
the XGB silent kernel death observed 2026-04-20.

Each trial runs ``XGBClassifier.fit()`` in a subprocess so a silent
crash kills only that child, not the bisector. We progressively halve
the set of categorical features (and then, once a culprit subset is
found, halve the rows) until we have a minimal reproducer.

Usage
-----
    python -m mlframe.profiling.bisect_xgb_prodcrash \
        --parquet R:\Data\Upwork\dataframes\PRODUCTION\jobs_details.parquet

Outputs (in the current working directory):
    bisect_xgb_prodcrash_log.txt   — full transcript of each trial
    bisect_xgb_prodcrash_repro.py  — minimal standalone reproducer (only
                                     produced if a crashing subset is
                                     successfully isolated)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# -------------------------------------------------------------------------
# Pipeline used by BOTH worker and orchestrator's baseline probe.
# Matches the user's Jupyter workflow:
#   - cast Float64->Float32, Utf8->Categorical
#   - sort by job_posted_at
#   - drop user-specified columns + the 6 drift-suspect ones
#   - extract cl_act_total_hired >= 1 as binary target
#   - add datetime features (hour/day/weekday/month from job_posted_at)
#   - drop job_posted_at, cl_id, uid, job_status, cl_act_total_hired
#   - promote columns with n_unique>300 to "text" (drop them — XGB doesn't
#     use text_features anyway in that tier)
#   - time-ordered split 80/10/10
#   - fill_null('__MISSING__') on cat columns that have nulls in ANY split
#   - feed to XGB as-is with pl.Categorical (no Enum alignment)
# -------------------------------------------------------------------------

DEFAULT_DROP_COLUMNS = [
    "uid", "job_posted_at", "job_status", "cl_id",
    "_raw_countries", "_raw_languages", "_raw_tags",
    "job_post_source", "job_post_device", "job_post_flow_type",
]

TARGET_COLUMN = "cl_act_total_hired"
TARGET_THRESHOLD = 1
TIMESTAMP_COLUMN = "job_posted_at"
HIGH_CARD_TEXT_THRESHOLD = 300  # matches mlframe's auto_detect default


def load_and_prepare(parquet_path: str, drop_cols: Sequence[str]):
    """Shared prep used by worker and baseline. Returns
    (train_df, val_df, test_df, y_train, y_val, y_test, cat_features).
    """
    import polars as pl
    import numpy as np

    df = pl.read_parquet(parquet_path)
    df = (
        df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
          .with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
          .sort(TIMESTAMP_COLUMN)
    )

    # Datetime features, same as SimpleFeaturesAndTargetsExtractor adds.
    df = df.with_columns([
        pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Int8).alias("hour"),
        pl.col(TIMESTAMP_COLUMN).dt.day().cast(pl.Int8).alias("day"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col(TIMESTAMP_COLUMN).dt.month().cast(pl.Int8).alias("month"),
    ])

    # Extract target before dropping its column. cl_act_total_hired can
    # be null — comparing null with >= 1 returns null, which propagates
    # through cast(Int8) and ends up as NaN in numpy, which makes XGB
    # choke with "Invalid classes ... got [0. 1. nan]". mlframe's
    # extractor handles this via classification_lower_thresholds; here
    # we fill_null(0) so null is treated as below-threshold = 0.
    target_series = df[TARGET_COLUMN].fill_null(0) >= TARGET_THRESHOLD
    target = target_series.cast(pl.Int8).to_numpy()

    to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(to_drop)

    # High-card text promotion (mimic mlframe auto_detect): drop columns
    # whose cardinality > threshold — XGB can't do anything with them.
    to_drop_text = []
    for c, dt in df.schema.items():
        if dt == pl.Categorical or dt == pl.Utf8:
            if df[c].n_unique() > HIGH_CARD_TEXT_THRESHOLD:
                to_drop_text.append(c)
    if to_drop_text:
        df = df.drop(to_drop_text)

    # Time-ordered 80/10/10 split.
    n = df.height
    n_test = int(n * 0.10)
    n_val = int(n * 0.10)
    n_train = n - n_val - n_test
    train_df = df[:n_train]
    val_df   = df[n_train : n_train + n_val]
    test_df  = df[n_train + n_val :]
    y_train = target[:n_train]
    y_val   = target[n_train : n_train + n_val]
    y_test  = target[n_train + n_val :]

    # Cat features after drops.
    cat_features = [c for c, dt in df.schema.items() if dt == pl.Categorical]

    # Round-17 fill_null on union of nullable cats across splits.
    from mlframe.training.trainer import (
        _polars_nullable_categorical_cols,
        _polars_fill_null_in_categorical,
    )
    tr_n = set(_polars_nullable_categorical_cols(train_df, cat_features=cat_features))
    v_n  = set(_polars_nullable_categorical_cols(val_df,   cat_features=cat_features))
    te_n = set(_polars_nullable_categorical_cols(test_df,  cat_features=cat_features))
    union = sorted(tr_n | v_n | te_n)
    if union:
        train_df = _polars_fill_null_in_categorical(train_df, union)
        val_df   = _polars_fill_null_in_categorical(val_df,   union)
        test_df  = _polars_fill_null_in_categorical(test_df,  union)

    return train_df, val_df, test_df, y_train, y_val, y_test, cat_features


# -------------------------------------------------------------------------
# Worker: runs one fit attempt. Exit 0 = passed, != 0 = crashed.
# -------------------------------------------------------------------------

def worker(state_file: str) -> int:
    # Suppress WER popup so kernel/process exits cleanly on silent kill.
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
        except Exception:
            pass
    try:
        import faulthandler; faulthandler.enable(all_threads=True)
    except Exception:
        pass

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    import polars as pl
    from xgboost import XGBClassifier

    t0 = time.perf_counter()
    train_df, val_df, test_df, y_tr, y_v, y_te, all_cat = load_and_prepare(
        state["parquet"], state["drop_cols"],
    )
    print(f"[worker] prepared frame in {time.perf_counter()-t0:.1f}s, "
          f"shape={train_df.shape}, all_cat={len(all_cat)}", flush=True)

    # Apply column subset (drop cats we're NOT testing).
    keep_cat = set(state["keep_cat_cols"])
    drop_cat = [c for c in all_cat if c not in keep_cat]
    if drop_cat:
        train_df = train_df.drop(drop_cat)
        val_df   = val_df.drop(drop_cat)

    # Apply row slice (if any).
    if state.get("row_limit") is not None:
        lim = state["row_limit"]
        train_df = train_df[:lim]
        y_tr = y_tr[:lim]
        print(f"[worker] row-sliced to train={lim}", flush=True)

    print(f"[worker] fit shape train={train_df.shape}, val={val_df.shape}, "
          f"kept cat={len(keep_cat)}: {sorted(keep_cat)}", flush=True)

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
    t0 = time.perf_counter()
    model.fit(train_df, y_tr, eval_set=[(val_df, y_v)], verbose=False)
    print(f"[worker] FIT_OK in {time.perf_counter()-t0:.1f}s, "
          f"best_iter={model.best_iteration}", flush=True)
    return 0


# -------------------------------------------------------------------------
# Orchestrator: bisects.
# -------------------------------------------------------------------------

OUTCOME_PASSED = "passed"
OUTCOME_CRASHED = "crashed"  # exit != 0, no Python traceback => silent kill
OUTCOME_RAISED = "raised"    # Python exception from XGB (e.g. unseen cat)
OUTCOME_TIMEOUT = "timeout"


def run_trial(
    parquet: str, drop_cols: Sequence[str],
    keep_cat_cols: Sequence[str], row_limit: Optional[int] = None,
    timeout: int = 900, log_prefix: str = "",
) -> Tuple[str, str]:
    """Returns (outcome, tail_of_stdout). Subprocess-isolated."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
        pickle.dump({
            "parquet": parquet,
            "drop_cols": list(drop_cols),
            "keep_cat_cols": list(keep_cat_cols),
            "row_limit": row_limit,
        }, f)
        state_file = f.name

    cmd = [sys.executable, __file__, "--worker", state_file]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        os.unlink(state_file)
        return OUTCOME_TIMEOUT, out[-2000:]
    os.unlink(state_file)

    out = proc.stdout + "\n" + proc.stderr
    tail = out[-2000:]
    elapsed = time.perf_counter() - t0

    # Outcome classification
    if proc.returncode == 0 and "FIT_OK" in out:
        outcome = OUTCOME_PASSED
    elif "XGBoostError" in out or "Traceback" in out:
        outcome = OUTCOME_RAISED
    else:
        # Non-zero exit without Python traceback => silent native kill.
        outcome = OUTCOME_CRASHED

    print(f"{log_prefix}trial keep={len(keep_cat_cols)} cats, rows={row_limit or 'full'} "
          f"-> {outcome} in {elapsed:.1f}s (rc={proc.returncode})", flush=True)
    return outcome, tail


def bisect_columns(
    parquet: str, drop_cols: Sequence[str], all_cat_cols: Sequence[str],
) -> List[str]:
    """Binary bisection over cat columns. Returns minimal crashing set.
    If neither half crashes alone, reports an interaction and returns
    the larger set."""
    print(f"\n=== Column bisection over {len(all_cat_cols)} cat features ===",
          flush=True)

    # Baseline sanity.
    outcome, tail = run_trial(parquet, drop_cols, all_cat_cols,
                              log_prefix="[baseline] ")
    if outcome != OUTCOME_CRASHED:
        print(f"[baseline] expected CRASHED (silent kill), got {outcome!r}.",
              flush=True)
        print(f"[baseline] tail of trial stdout/stderr:\n{tail}", flush=True)
        if outcome == OUTCOME_RAISED:
            print(f"[baseline] trial raised a Python exception — this is "
                  f"NOT the silent kernel death we're trying to reproduce. "
                  f"Either (a) the bisector's data prep diverges from "
                  f"mlframe's, or (b) the bug surfaces as a Python exception "
                  f"in a subprocess context (different from the notebook "
                  f"context where it silently kills). Fix (a) before "
                  f"continuing.", flush=True)
        else:
            print(f"[baseline] trial passed — no crash to bisect.",
                  flush=True)
        return []

    current = sorted(all_cat_cols)
    while len(current) > 1:
        mid = len(current) // 2
        left, right = current[:mid], current[mid:]
        out_left, _ = run_trial(parquet, drop_cols, left,
                                log_prefix="  [left]  ")
        if out_left == OUTCOME_CRASHED:
            current = left
            print(f"  -> narrowed to left half ({len(current)}): {current}",
                  flush=True)
            continue
        out_right, _ = run_trial(parquet, drop_cols, right,
                                 log_prefix="  [right] ")
        if out_right == OUTCOME_CRASHED:
            current = right
            print(f"  -> narrowed to right half ({len(current)}): {current}",
                  flush=True)
            continue
        # Neither half crashes alone — interaction needed.
        print(f"  -> interaction: neither half crashes; culprit needs >=2 "
              f"columns. Reporting current set of {len(current)}: {current}",
              flush=True)
        return current

    print(f"=== SINGLE CULPRIT: {current} ===", flush=True)
    return current


def bisect_rows(
    parquet: str, drop_cols: Sequence[str], cat_cols: Sequence[str],
    full_n_train: int,
) -> Optional[int]:
    """Given the crashing column set, find minimal n_train that still
    crashes (binary search downward). Returns the smallest n_train
    that reproduces."""
    print(f"\n=== Row bisection (min n_train that still crashes) ===",
          flush=True)
    # Confirm full scale crashes.
    out, _ = run_trial(parquet, drop_cols, cat_cols,
                       row_limit=full_n_train, log_prefix="[rows full] ")
    if out != OUTCOME_CRASHED:
        print(f"[rows] full n_train={full_n_train:_} did not crash ({out}). "
              f"Skipping row bisect.", flush=True)
        return None

    lo, hi = 100_000, full_n_train
    best_crash = hi
    while hi - lo > 200_000:
        mid = (lo + hi) // 2
        out, _ = run_trial(parquet, drop_cols, cat_cols, row_limit=mid,
                           log_prefix="  [rows]  ")
        if out == OUTCOME_CRASHED:
            best_crash = mid
            hi = mid
        else:
            lo = mid
    print(f"=== MIN CRASHING n_train ~= {best_crash:_} ===", flush=True)
    return best_crash


def write_reproducer(
    out_path: Path, parquet: str, drop_cols: Sequence[str],
    culprit_cats: Sequence[str], min_rows: Optional[int],
) -> None:
    body = f'''"""Auto-generated minimal reproducer for the XGB silent kill
observed on prod_jobsdetails 2026-04-20.

Bisection result:
  parquet path:       {parquet!r}
  dropped columns:    {list(drop_cols)!r}
  culprit cat set:    {list(culprit_cats)!r}
  min rows to crash:  {min_rows!r}
"""
import sys, time, polars as pl, numpy as np
from xgboost import XGBClassifier

PARQUET = {parquet!r}
DROP_COLS = {list(drop_cols)!r}
CULPRIT_CATS = {list(culprit_cats)!r}
ROW_LIMIT = {min_rows!r}

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception:
    pass


def main():
    df = pl.read_parquet(PARQUET)
    df = (df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
            .with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
            .sort("job_posted_at"))
    df = df.with_columns([
        pl.col("job_posted_at").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("job_posted_at").dt.day().cast(pl.Int8).alias("day"),
        pl.col("job_posted_at").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("job_posted_at").dt.month().cast(pl.Int8).alias("month"),
    ])
    # fill_null(0) before threshold — matches what the bisector does
    # in load_and_prepare(). Otherwise null propagates to NaN and XGB
    # raises 'Label contains NaN' instead of the real crash.
    target = (df["cl_act_total_hired"].fill_null(0) >= 1).cast(pl.Int8).to_numpy()
    df = df.drop([c for c in DROP_COLS if c in df.columns])

    # Keep only culprit cat columns + all numerics.
    all_cat = [c for c, dt in df.schema.items() if dt == pl.Categorical]
    drop_other_cat = [c for c in all_cat if c not in CULPRIT_CATS]
    df = df.drop(drop_other_cat)

    n = df.height
    n_test = int(n * 0.10); n_val = int(n * 0.10); n_train = n - n_val - n_test
    train = df[:n_train]; val = df[n_train:n_train+n_val]
    y_tr = target[:n_train]; y_v = target[n_train:n_train+n_val]

    if ROW_LIMIT is not None:
        train = train[:ROW_LIMIT]
        y_tr = y_tr[:ROW_LIMIT]

    print(f"train={{train.shape}}, val={{val.shape}}, cats={{CULPRIT_CATS}}")

    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1, max_cat_to_onehot=1,
        max_cat_threshold=100, early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    model.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
    print(f"FIT_OK in {{time.perf_counter()-t0:.1f}}s (bug did NOT reproduce)")


if __name__ == "__main__":
    main()
'''
    out_path.write_text(body, encoding="utf-8")
    print(f"\n=== Wrote minimal reproducer to {out_path} ===", flush=True)


def main_orchestrator():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=str,
                    help=r"Path to jobs_details.parquet")
    ap.add_argument("--drop-cols", nargs="*", default=None,
                    help="Columns to drop up-front. Default: extractor "
                         "drops + 6 drift suspects.")
    ap.add_argument("--trial-timeout", type=int, default=900,
                    help="Per-trial timeout in seconds.")
    args = ap.parse_args()

    drop_cols = args.drop_cols if args.drop_cols is not None else DEFAULT_DROP_COLUMNS

    # Probe full data once to get the current cat_features list +
    # n_train (needed for row bisect).
    import polars as pl
    print(f"[orchestrator] probing full frame to enumerate cat features...",
          flush=True)
    train_df, val_df, test_df, y_tr, y_v, y_te, cat_features = (
        load_and_prepare(args.parquet, drop_cols)
    )
    n_train = train_df.height
    print(f"[orchestrator] train shape={train_df.shape}, "
          f"cat_features ({len(cat_features)}): {cat_features}", flush=True)
    # Free memory before subprocess spawning.
    del train_df, val_df, test_df, y_tr, y_v, y_te

    # Bisect columns.
    culprit_cats = bisect_columns(args.parquet, drop_cols, cat_features)
    if not culprit_cats:
        print("[orchestrator] no culprit found — baseline did not crash. "
              "Nothing to reproduce.", flush=True)
        return

    # Bisect rows.
    min_rows = bisect_rows(args.parquet, drop_cols, culprit_cats, n_train)

    # Write reproducer.
    out = Path("bisect_xgb_prodcrash_repro.py").absolute()
    write_reproducer(out, args.parquet, drop_cols, culprit_cats, min_rows)


if __name__ == "__main__":
    # Dual-mode: worker (--worker STATE.pkl) or orchestrator.
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        rc = worker(sys.argv[2])
        sys.exit(rc)
    main_orchestrator()
