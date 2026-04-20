r"""Minimize the reproducer: find the smallest subset of columns that
still triggers the XGB silent kill on prod data.

Two axes:

**Phase 1 — string columns (cache-pollution sources).**
  Prod casts many Utf8 columns to pl.Categorical simultaneously.
  The non-``category`` string columns are then dropped before XGB —
  their only purpose is to pollute the polars StringCache, giving
  ``category`` sparse physical codes. We bisect: which are actually
  needed? Criterion = ``category``'s ``physical_codes_range[1]``
  stays above a sparseness threshold (default 10_000). No XGB
  involvement — fast, in-process.

**Phase 2 — numeric columns (fit inputs).**
  With the minimal string set fixed, we bisect the numerics.
  Criterion = XGB still silently kills. Each trial runs in a
  subprocess to isolate the crash.

Final output: minimum ``{string cols, numeric cols}`` pair. Together
with ``category`` and ~200k rows this should be the smallest
attachable reproducer for an xgboost issue.

Usage:
    python -m mlframe.profiling.minimize_xgb_crash_cols --parquet "R:\\..\\jobs_details.parquet"

Approximate runtime on prod box (128GB): 5-10 min Phase 1 (in-process,
just frame inspection), 30-60 min Phase 2 (subprocess XGB fits).
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
from typing import List, Sequence, Tuple

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
DEFAULT_DROP_COLUMNS = [
    "uid", "job_posted_at", "job_status", "cl_id",
]
ROW_LIMIT = 211_168
SPARSENESS_THRESHOLD = 10_000  # Phase 1 OK if max physical code > this


# ---------------------------------------------------------------------------
# Phase 1 — in-process check of category's physical_codes_range
# ---------------------------------------------------------------------------

def category_codes_max(parquet_path: str, string_cols_to_cast: Sequence[str]) -> int:
    """Load parquet, cast all specified string-like cols + ``category`` via
    ``.cast(pl.String).cast(pl.Categorical)`` in a single ``.with_columns()``
    so they all participate in a shared cache. Report max physical code
    for ``category``. Returns -1 on error.

    The double cast (`→ String → Categorical`) is needed when ``category``
    is ALREADY Categorical in the parquet — a straight
    ``.cast(pl.Categorical)`` on an already-Categorical column is a
    no-op that keeps the original per-Series dict. Routing through
    String forces polars to re-materialise the Categorical alongside
    the other columns in the shared-cache batch.
    """
    import polars as pl
    try:
        df = pl.read_parquet(parquet_path)
        df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        to_cast = list(set(string_cols_to_cast) | {CULPRIT_CAT})
        existing = [c for c, dt in df.schema.items()
                    if c in to_cast and dt in (pl.Utf8, pl.String, pl.Categorical)]
        if existing:
            df = df.with_columns([
                pl.col(c).cast(pl.String).cast(pl.Categorical) for c in existing
            ])
        if CULPRIT_CAT not in df.columns:
            return -1
        return int(df[CULPRIT_CAT].to_physical().max())
    except Exception as e:
        print(f"  [codes_max] error: {e}", flush=True)
        return -1


def enumerate_string_cols(parquet_path: str) -> List[str]:
    """Return all Utf8/String/Categorical columns except 'category' (we
    ALWAYS include category). Includes existing Categoricals because
    round-through-String casts them with the shared-cache batch."""
    import polars as pl
    df = pl.read_parquet(parquet_path, n_rows=1000)
    out = []
    print(f"  parquet schema scan:", flush=True)
    for c, dt in df.schema.items():
        if dt in (pl.Utf8, pl.String, pl.Categorical):
            print(f"    {c}: {dt}", flush=True)
            if c != CULPRIT_CAT:
                out.append(c)
    return out


def bisect_string_cols(parquet_path: str, all_strings: Sequence[str]) -> List[str]:
    """Binary-bisect: find minimum set of string cols whose cache pollution
    keeps category's max physical code above SPARSENESS_THRESHOLD."""
    print(f"\n=== Phase 1: bisecting {len(all_strings)} string columns ===",
          flush=True)
    baseline = category_codes_max(parquet_path, all_strings)
    print(f"  baseline max code with all {len(all_strings)} strings: {baseline}",
          flush=True)
    if baseline < SPARSENESS_THRESHOLD:
        print(f"  baseline below threshold {SPARSENESS_THRESHOLD} — no sparseness "
              f"to preserve, skipping phase 1", flush=True)
        return list(all_strings)
    # Also check with zero string cols: should give compact code (cat only).
    zero = category_codes_max(parquet_path, [])
    print(f"  zero-string baseline: max code={zero} (compact target)", flush=True)

    current = list(all_strings)
    while len(current) > 1:
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]
        t0 = time.perf_counter()
        left_max = category_codes_max(parquet_path, left)
        print(f"  [left  half={len(left)}] max code={left_max}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
        if left_max >= SPARSENESS_THRESHOLD:
            current = left
            continue
        t0 = time.perf_counter()
        right_max = category_codes_max(parquet_path, right)
        print(f"  [right half={len(right)}] max code={right_max}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
        if right_max >= SPARSENESS_THRESHOLD:
            current = right
            continue
        # Neither half alone is enough — the scatter comes from interaction.
        # We'd need multiple string cols together. Combine halves via LOO:
        # remove one col at a time, keep if still above threshold.
        print(f"  interaction: neither half alone enough; entering LOO prune",
              flush=True)
        return loo_prune_strings(parquet_path, current)
    # Down to 1 — check it's still enough.
    if category_codes_max(parquet_path, current) >= SPARSENESS_THRESHOLD:
        print(f"  SINGLE STRING CULPRIT: {current}", flush=True)
    return current


def loo_prune_strings(parquet_path: str, cols: List[str]) -> List[str]:
    """Leave-one-out pruning: remove cols that aren't needed for sparseness."""
    current = list(cols)
    i = 0
    while i < len(current):
        candidate = current[:i] + current[i+1:]
        t0 = time.perf_counter()
        mx = category_codes_max(parquet_path, candidate)
        if mx >= SPARSENESS_THRESHOLD:
            print(f"  LOO prune: {current[i]!r} unneeded (max={mx})  "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            current = candidate
        else:
            print(f"  LOO keep:  {current[i]!r} needed (without={mx})  "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            i += 1
    return current


# ---------------------------------------------------------------------------
# Phase 2 — subprocess XGB fit bisection over numeric cols
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

    import polars as pl
    import numpy as np
    from xgboost import XGBClassifier

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    parquet_path = state["parquet"]
    string_cols  = state["string_cols"]
    numeric_cols = state["numeric_cols"]

    t0 = time.perf_counter()
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    # Route through String so Categoricals join the shared-cache batch.
    to_cast = list(set(string_cols) | {CULPRIT_CAT})
    existing = [c for c, dt in df.schema.items()
                if c in to_cast and dt in (pl.Utf8, pl.String, pl.Categorical)]
    if existing:
        df = df.with_columns([
            pl.col(c).cast(pl.String).cast(pl.Categorical) for c in existing
        ])
    df = df.sort(TIMESTAMP_COLUMN)
    # Datetime features.
    df = df.with_columns([
        pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Int8).alias("hour"),
        pl.col(TIMESTAMP_COLUMN).dt.day().cast(pl.Int8).alias("day"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col(TIMESTAMP_COLUMN).dt.month().cast(pl.Int8).alias("month"),
    ])
    target = (df[TARGET_COLUMN].fill_null(0) >= TARGET_THRESHOLD).cast(pl.Int8).to_numpy()

    # Keep ONLY: category + specified numeric cols + datetime features.
    dt_feats = ["hour", "day", "weekday", "month"]
    keep = [CULPRIT_CAT] + list(numeric_cols) + [c for c in dt_feats if c in df.columns]
    keep_existing = [c for c in keep if c in df.columns]
    df = df.select(keep_existing)

    # Fill_null on category.
    if df[CULPRIT_CAT].null_count() > 0:
        df = df.with_columns(pl.col(CULPRIT_CAT).fill_null("__MISSING__"))

    n = df.height
    n_test = int(n * 0.10); n_val = int(n * 0.10); n_train = n - n_val - n_test
    train = df[:n_train][:ROW_LIMIT]
    val   = df[n_train : n_train + n_val]
    y_tr = target[:n_train][:ROW_LIMIT]
    y_v  = target[n_train : n_train + n_val]

    mx = int(train[CULPRIT_CAT].to_physical().max())
    print(f"[worker] shape={train.shape}, category physical_codes_max={mx}, "
          f"prep={time.perf_counter()-t0:.1f}s", flush=True)

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


def run_trial(parquet: str, string_cols: Sequence[str],
              numeric_cols: Sequence[str], timeout: int = 900,
              log_prefix: str = "") -> str:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
        pickle.dump({
            "parquet": parquet,
            "string_cols": list(string_cols),
            "numeric_cols": list(numeric_cols),
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
    out = proc.stdout + "\n" + proc.stderr
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0 and "FIT_OK" in out:
        outcome = OUTCOME_PASSED
    elif "Traceback" in out or "Error" in out.split("\n")[-5:][0] if out.split("\n") else "":
        outcome = OUTCOME_RAISED
    elif proc.returncode != 0:
        outcome = OUTCOME_CRASHED
    else:
        outcome = OUTCOME_PASSED
    print(f"{log_prefix}trial strings={len(string_cols)}, numerics={len(numeric_cols)} "
          f"-> {outcome} (rc={proc.returncode}) in {elapsed:.1f}s", flush=True)
    return outcome


def enumerate_numeric_cols(parquet_path: str) -> List[str]:
    import polars as pl
    df = pl.read_parquet(parquet_path, n_rows=1000)
    # Consider numeric = everything except strings, target, timestamp, and drop cols.
    numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Boolean)
    excluded = set([TARGET_COLUMN, TIMESTAMP_COLUMN, CULPRIT_CAT] + DEFAULT_DROP_COLUMNS)
    return [c for c, dt in df.schema.items()
            if dt in numeric_dtypes and c not in excluded]


def bisect_numeric_cols(parquet: str, strings: Sequence[str],
                        all_numerics: Sequence[str]) -> List[str]:
    """Binary-bisect numerics keeping strings fixed. Find minimum set
    that still crashes (outcome = OUTCOME_CRASHED)."""
    print(f"\n=== Phase 2: bisecting {len(all_numerics)} numeric columns ===",
          flush=True)
    base = run_trial(parquet, strings, all_numerics, log_prefix="[base]   ")
    if base != OUTCOME_CRASHED:
        print(f"  baseline did not crash (got {base}) — cannot bisect numerics",
              flush=True)
        return list(all_numerics)
    current = list(all_numerics)
    while len(current) > 1:
        mid = len(current) // 2
        left = current[:mid]; right = current[mid:]
        left_r = run_trial(parquet, strings, left, log_prefix="  [left]  ")
        if left_r == OUTCOME_CRASHED:
            current = left
            continue
        right_r = run_trial(parquet, strings, right, log_prefix="  [right] ")
        if right_r == OUTCOME_CRASHED:
            current = right
            continue
        print(f"  interaction: neither half of numerics alone crashes; "
              f"keeping {len(current)}: {current}", flush=True)
        return current
    print(f"=== SINGLE NUMERIC CULPRIT: {current} ===", flush=True)
    return current


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=str)
    ap.add_argument("--phase1-only", action="store_true",
                    help="Skip phase 2; just report minimum string cols.")
    ap.add_argument("--strings", nargs="*", default=None,
                    help="Pre-specified string cols (skip phase 1).")
    ap.add_argument("--trial-timeout", type=int, default=900)
    args = ap.parse_args()

    all_strings = enumerate_string_cols(args.parquet)
    print(f"Found {len(all_strings)} string columns (excluding 'category'): "
          f"{all_strings}", flush=True)
    all_numerics = enumerate_numeric_cols(args.parquet)
    print(f"Found {len(all_numerics)} numeric columns: {all_numerics}",
          flush=True)

    # Phase 1.
    if args.strings is not None:
        print(f"\n=== Skipping Phase 1, using provided strings: {args.strings} ===",
              flush=True)
        min_strings = args.strings
    else:
        min_strings = bisect_string_cols(args.parquet, all_strings)
        print(f"\n=== Phase 1 result: {len(min_strings)} string cols needed ===",
              flush=True)
        print(f"  {min_strings}", flush=True)

    if args.phase1_only:
        return

    # Phase 2.
    min_numerics = bisect_numeric_cols(args.parquet, min_strings, all_numerics)
    print(f"\n=== Phase 2 result: {len(min_numerics)} numeric cols needed ===",
          flush=True)
    print(f"  {min_numerics}", flush=True)

    print(f"\n=== FINAL MINIMAL REPRODUCER SHAPE ===", flush=True)
    print(f"  Parquet: {args.parquet}", flush=True)
    print(f"  Cast these Utf8 cols to Categorical together:  "
          f"{min_strings + [CULPRIT_CAT]}", flush=True)
    print(f"  After cast, keep only:  category + {min_numerics}", flush=True)
    print(f"  Row slice: first {ROW_LIMIT:_} rows (time-ordered)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    main()
