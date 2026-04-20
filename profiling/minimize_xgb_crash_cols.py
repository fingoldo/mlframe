r"""Minimize the reproducer: find the smallest subset of columns that
still triggers the XGB silent kill on prod data.

Uses the FULL production pipeline (not just cast): load, Float64→32,
Utf8→Categorical, sort, datetime features, target extract, drop
user-drop cols, text-promotion drops, keep culprit+numerics,
fill_null('__MISSING__'), time-ordered split, slice to ROW_LIMIT.

The ``fill_null`` step is the one that re-resolves ``category``'s
physical codes through the global StringCache (confirmed 2026-04-20
by ``trace_category_codes.py``). Without it, codes stay compact.

Two axes:

**Phase 1 — string columns (cache-pollution sources).**
  Bisect the set of Utf8/Categorical columns whose presence during
  the Utf8→Categorical cast produces the ``[103, 3_287_945]``-style
  sparse codes on ``category`` after fill_null. Check in-process,
  no XGB.

**Phase 2 — numeric columns (fit inputs).**
  With the minimal string set fixed, bisect numeric columns by
  running XGB.fit in a subprocess (silent kill-tolerant). Find the
  minimum set that still crashes.

Final output: minimum ``{string cols to cast, numeric cols to keep}``
pair. Together with ``category``, fill_null, and ~211k rows this is
the smallest attachable reproducer for an xgboost issue.

Usage:
    python -m mlframe.profiling.minimize_xgb_crash_cols --parquet "R:\\..\\jobs_details.parquet"

Approximate runtime on 128GB prod box: 10-20 min Phase 1 (in-process
full pipeline), 30-90 min Phase 2 (subprocess XGB fits).
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    except Exception:
        pass

TARGET_COLUMN = "cl_act_total_hired"
TARGET_THRESHOLD = 1
TIMESTAMP_COLUMN = "job_posted_at"
CULPRIT_CAT = "category"
# "Drop cols" = user-specified drops (uid/cl_id/job_status/etc); NOT
# the candidate strings — those we're bisecting.
USER_DROP_COLUMNS = ["uid", "job_posted_at", "job_status", "cl_id"]
ROW_LIMIT = 211_168

# Phase 1 criterion: category's max physical code after the full
# pipeline must stay above this. Prod crash was at max=3_287_945;
# 10_000 is two orders of magnitude clear of the ~500 compact baseline.
SPARSENESS_THRESHOLD = 10_000


# ---------------------------------------------------------------------------
# Shared pipeline: full prep mimicking prod, minus the Utf8 columns we
# choose to exclude from the simultaneous cast.
# ---------------------------------------------------------------------------

def prepare_frame(
    parquet_path: str,
    strings_to_cast: Sequence[str],
    keep_numeric: Optional[Sequence[str]] = None,
):
    """Return (train, val, y_train, y_val). Applies full prod pipeline.

    - ``strings_to_cast``: Utf8/Categorical columns routed through
      ``.cast(String).cast(Categorical)`` in the SAME with_columns()
      call as ``category``. This puts them in a shared-cache batch
      so ``category`` can get scattered codes after fill_null.
    - ``keep_numeric``: if None, keep all numerics. If provided, keep
      only the named subset.

    Columns NOT in strings_to_cast and NOT in keep_numeric are dropped
    before any preprocessing (keeps memory down for big parquets).
    """
    import polars as pl
    import numpy as np

    df = pl.read_parquet(parquet_path)

    # Identify columns we need: user drops + target + timestamp + category
    # + strings_to_cast + (keep_numeric or all numerics).
    reserved = set(USER_DROP_COLUMNS) | {TARGET_COLUMN, TIMESTAMP_COLUMN, CULPRIT_CAT}
    reserved |= set(strings_to_cast)

    # Determine numeric cols (everything not String-like, not reserved).
    numeric_dtypes = (
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Boolean,
    )
    all_numerics = [c for c, dt in df.schema.items()
                    if dt in numeric_dtypes and c not in reserved]
    if keep_numeric is None:
        kept_numerics = all_numerics
    else:
        kept_numerics = [c for c in keep_numeric if c in all_numerics]

    keep_all = list(reserved | set(kept_numerics))
    keep_existing = [c for c in keep_all if c in df.columns]
    df = df.select(keep_existing)

    # Prod prep begins.
    df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    # Cast selected string-like columns + category through String to force
    # shared-cache re-materialization.
    cast_set = set(strings_to_cast) | {CULPRIT_CAT}
    existing_cast = [c for c, dt in df.schema.items()
                     if c in cast_set and dt in (pl.Utf8, pl.String, pl.Categorical)]
    if existing_cast:
        df = df.with_columns([
            pl.col(c).cast(pl.String).cast(pl.Categorical) for c in existing_cast
        ])

    df = df.sort(TIMESTAMP_COLUMN)
    df = df.with_columns([
        pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Int8).alias("hour"),
        pl.col(TIMESTAMP_COLUMN).dt.day().cast(pl.Int8).alias("day"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col(TIMESTAMP_COLUMN).dt.month().cast(pl.Int8).alias("month"),
    ])
    target = (df[TARGET_COLUMN].fill_null(0) >= TARGET_THRESHOLD).cast(pl.Int8).to_numpy()

    # Drop user-drop cols.
    to_drop = [c for c in USER_DROP_COLUMNS if c in df.columns]
    df = df.drop(to_drop)

    # Text-promotion drops: any remaining Categorical with n_unique > 300.
    HIGH_CARD = 300
    to_drop_text = [c for c, dt in df.schema.items()
                    if dt in (pl.Categorical, pl.Utf8) and df[c].n_unique() > HIGH_CARD]
    if to_drop_text:
        df = df.drop(to_drop_text)

    # Drop other Categoricals (non-culprit) — XGB only sees category + numerics.
    other_cats = [c for c, dt in df.schema.items()
                  if dt == pl.Categorical and c != CULPRIT_CAT]
    if other_cats:
        df = df.drop(other_cats)

    # Round-17 fill_null — THIS is what re-resolves category's codes.
    if CULPRIT_CAT in df.columns and df[CULPRIT_CAT].null_count() > 0:
        df = df.with_columns(pl.col(CULPRIT_CAT).fill_null("__MISSING__"))

    # Split 80/10/10 time-ordered.
    n = df.height
    n_test = int(n * 0.10); n_val = int(n * 0.10); n_train = n - n_val - n_test
    train = df[:n_train][:ROW_LIMIT]
    val   = df[n_train : n_train + n_val]
    y_tr = target[:n_train][:ROW_LIMIT]
    y_v  = target[n_train : n_train + n_val]
    return train, val, y_tr, y_v


def category_max_code(parquet_path: str, strings_to_cast: Sequence[str]) -> int:
    """Run full pipeline, return category's physical_codes_max in train."""
    try:
        train, _, _, _ = prepare_frame(parquet_path, strings_to_cast)
        if CULPRIT_CAT not in train.columns:
            return -1
        return int(train[CULPRIT_CAT].to_physical().max())
    except Exception as e:
        print(f"  [codes_max] error: {e}", flush=True)
        return -1


# ---------------------------------------------------------------------------
# Phase 1 — string cols bisection
# ---------------------------------------------------------------------------

def enumerate_string_cols(parquet_path: str) -> List[str]:
    """Return all Utf8/String/Categorical columns except 'category'."""
    import polars as pl
    df = pl.read_parquet(parquet_path, n_rows=1000)
    out = []
    print(f"  parquet schema scan (string-like cols):", flush=True)
    for c, dt in df.schema.items():
        if dt in (pl.Utf8, pl.String, pl.Categorical):
            print(f"    {c}: {dt}", flush=True)
            if c != CULPRIT_CAT:
                out.append(c)
    return out


def bisect_string_cols(parquet_path: str, all_strings: Sequence[str]) -> List[str]:
    print(f"\n=== Phase 1: bisecting {len(all_strings)} string columns ===",
          flush=True)
    t0 = time.perf_counter()
    baseline = category_max_code(parquet_path, all_strings)
    print(f"  baseline max code with all {len(all_strings)} strings "
          f"(FULL pipeline incl fill_null): {baseline}  ({time.perf_counter()-t0:.1f}s)",
          flush=True)
    if baseline < SPARSENESS_THRESHOLD:
        print(f"  baseline below threshold {SPARSENESS_THRESHOLD} — nothing "
              f"sparse to bisect. Phase 1 returns all strings unchanged.", flush=True)
        return list(all_strings)
    zero = category_max_code(parquet_path, [])
    print(f"  zero-strings max code: {zero} (compact target if near 0)",
          flush=True)

    current = list(all_strings)
    while len(current) > 1:
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]
        t0 = time.perf_counter()
        left_max = category_max_code(parquet_path, left)
        print(f"  [left  {len(left)}] max code={left_max}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
        if left_max >= SPARSENESS_THRESHOLD:
            current = left
            continue
        t0 = time.perf_counter()
        right_max = category_max_code(parquet_path, right)
        print(f"  [right {len(right)}] max code={right_max}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
        if right_max >= SPARSENESS_THRESHOLD:
            current = right
            continue
        # Interaction: neither half alone. Pivot to leave-one-out.
        print(f"  interaction: neither half alone; LOO-pruning over "
              f"{len(current)} cols", flush=True)
        return loo_prune_strings(parquet_path, current)
    # Single col — verify.
    m = category_max_code(parquet_path, current)
    print(f"  single col {current[0]!r} gives max code={m}", flush=True)
    return current


def loo_prune_strings(parquet_path: str, cols: List[str]) -> List[str]:
    current = list(cols)
    i = 0
    while i < len(current):
        candidate = current[:i] + current[i+1:]
        t0 = time.perf_counter()
        m = category_max_code(parquet_path, candidate)
        if m >= SPARSENESS_THRESHOLD:
            print(f"  LOO drop {current[i]!r} (without: max={m})  "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            current = candidate
        else:
            print(f"  LOO keep {current[i]!r} (without: max={m})  "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            i += 1
    return current


# ---------------------------------------------------------------------------
# Phase 2 — subprocess-isolated XGB fit bisection over numeric cols
# ---------------------------------------------------------------------------

OUTCOME_PASSED = "passed"
OUTCOME_CRASHED = "crashed"
OUTCOME_RAISED = "raised"
OUTCOME_TIMEOUT = "timeout"


def worker(state_file: str) -> int:
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

    t0 = time.perf_counter()
    train, val, y_tr, y_v = prepare_frame(
        state["parquet"], state["strings"], state["numerics"],
    )
    print(f"[worker] prep in {time.perf_counter()-t0:.1f}s, "
          f"train={train.shape}, val={val.shape}", flush=True)
    if CULPRIT_CAT in train.columns:
        mx = int(train[CULPRIT_CAT].to_physical().max())
        print(f"[worker] category physical_codes_max={mx}", flush=True)

    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1,
        max_cat_to_onehot=1, max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    t0 = time.perf_counter()
    model.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
    print(f"[worker] FIT_OK in {time.perf_counter()-t0:.1f}s", flush=True)
    return 0


def run_trial(
    parquet: str, strings: Sequence[str], numerics: Sequence[str],
    timeout: int = 900, log_prefix: str = "",
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
        pickle.dump({
            "parquet": parquet,
            "strings": list(strings),
            "numerics": list(numerics),
        }, f)
        state_file = f.name
    cmd = [sys.executable, __file__, "--worker", state_file]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        os.unlink(state_file)
        return OUTCOME_TIMEOUT
    os.unlink(state_file)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0 and "FIT_OK" in out:
        outcome = OUTCOME_PASSED
    elif "Traceback" in out:
        outcome = OUTCOME_RAISED
    elif proc.returncode != 0:
        outcome = OUTCOME_CRASHED
    else:
        outcome = OUTCOME_PASSED
    print(f"{log_prefix}trial strings={len(strings)}, numerics={len(numerics)} "
          f"-> {outcome} (rc={proc.returncode}) in {elapsed:.1f}s", flush=True)
    return outcome


def enumerate_numeric_cols(parquet_path: str) -> List[str]:
    import polars as pl
    df = pl.read_parquet(parquet_path, n_rows=1000)
    numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Boolean)
    excluded = set([TARGET_COLUMN, TIMESTAMP_COLUMN, CULPRIT_CAT] + USER_DROP_COLUMNS)
    return [c for c, dt in df.schema.items()
            if dt in numeric_dtypes and c not in excluded]


def bisect_numeric_cols(
    parquet: str, strings: Sequence[str], all_numerics: Sequence[str],
    trial_timeout: int,
) -> List[str]:
    print(f"\n=== Phase 2: bisecting {len(all_numerics)} numeric columns "
          f"(full prod pipeline in subprocess) ===", flush=True)
    base = run_trial(parquet, strings, all_numerics, timeout=trial_timeout,
                     log_prefix="[base]   ")
    if base != OUTCOME_CRASHED:
        print(f"  baseline did not crash ({base}) — cannot bisect", flush=True)
        return list(all_numerics)

    # Short-circuit: maybe no numerics are needed at all.
    empty = run_trial(parquet, strings, [], timeout=trial_timeout,
                      log_prefix="[empty]  ")
    if empty == OUTCOME_CRASHED:
        print(f"  ZERO numerics needed — category alone + fill_null triggers "
              f"the crash. Returning []", flush=True)
        return []

    current = list(all_numerics)
    while len(current) > 1:
        mid = len(current) // 2
        left = current[:mid]; right = current[mid:]
        lr = run_trial(parquet, strings, left,  timeout=trial_timeout,
                       log_prefix="  [left]  ")
        if lr == OUTCOME_CRASHED:
            current = left
            continue
        rr = run_trial(parquet, strings, right, timeout=trial_timeout,
                       log_prefix="  [right] ")
        if rr == OUTCOME_CRASHED:
            current = right
            continue
        print(f"  interaction: neither numeric half alone crashes; returning "
              f"{len(current)}: {current}", flush=True)
        return current
    # Down to 1 — check if even fewer (zero additional) also crashes.
    # We already tested empty above, so if we got here via bisection,
    # current[0] alone might still not crash — verify.
    one_r = run_trial(parquet, strings, current, timeout=trial_timeout,
                     log_prefix="  [one]   ")
    if one_r == OUTCOME_CRASHED:
        print(f"=== SINGLE NUMERIC CULPRIT: {current} ===", flush=True)
    else:
        print(f"  single-numeric trial did NOT crash ({one_r}); "
              f"falling back to larger set. Interaction-driven.", flush=True)
    return current


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=str)
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--strings", nargs="*", default=None,
                    help="Override phase 1; use these strings in phase 2.")
    ap.add_argument("--trial-timeout", type=int, default=900)
    args = ap.parse_args()

    all_strings = enumerate_string_cols(args.parquet)
    print(f"Found {len(all_strings)} string-like columns (excluding 'category'): "
          f"{all_strings}", flush=True)
    all_numerics = enumerate_numeric_cols(args.parquet)
    print(f"Found {len(all_numerics)} numeric columns", flush=True)

    # Phase 1
    if args.strings is not None:
        print(f"\n=== Skipping Phase 1, using provided strings ===", flush=True)
        min_strings = args.strings
    else:
        min_strings = bisect_string_cols(args.parquet, all_strings)
        print(f"\n=== Phase 1 result: {len(min_strings)} string cols needed ===",
              flush=True)
        print(f"  {min_strings}", flush=True)

    if args.phase1_only:
        return

    # Phase 2
    min_numerics = bisect_numeric_cols(args.parquet, min_strings, all_numerics,
                                       trial_timeout=args.trial_timeout)
    print(f"\n=== Phase 2 result: {len(min_numerics)} numeric cols needed ===",
          flush=True)
    print(f"  {min_numerics}", flush=True)

    print(f"\n=== FINAL MINIMAL REPRODUCER SHAPE ===", flush=True)
    print(f"  Parquet: {args.parquet}", flush=True)
    print(f"  Cast through String then Categorical together: "
          f"{sorted(set(min_strings) | {CULPRIT_CAT})}", flush=True)
    print(f"  Keep only in XGB frame:  category + {min_numerics}", flush=True)
    print(f"  Apply fill_null('__MISSING__') on category (re-resolves codes)",
          flush=True)
    print(f"  Row slice: first {ROW_LIMIT:_} rows (time-ordered)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    main()
