r"""Binary-search the minimum n_train rows that still triggers the
silent kill in the minimal reproducer.

Pipeline (same as ``no_replace_check.py``):
  load skills_text + category + job_posted_at -> sort -> cast both
  through String -> Categorical -> drop skills_text -> fill_null on
  category -> XGB.fit on first ``n_train`` rows + ``n_val`` rows.

Subprocess-isolated trials so silent kill of one trial doesn't kill
the bisector.

Usage:
    python -m mlframe.profiling.bisect_xgb_min_rows --parquet "R:\\..\\jobs_details.parquet"

Defaults bracket the search at 211_168 (known crashing) and 1_000
(very small). log2(211_168) ≈ 18 levels; each trial ~30-90s ->
~10-30 min total.
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

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    except Exception:
        pass


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

    import numpy as np
    import polars as pl
    from xgboost import XGBClassifier

    with open(state_file, "rb") as f:
        state = pickle.load(f)
    parquet = state["parquet"]
    n_train = state["n_train"]
    n_val   = state["n_val"]

    t0 = time.perf_counter()
    df = pl.read_parquet(parquet, columns=["skills_text", "category", "job_posted_at"])
    df = df.sort("job_posted_at").drop("job_posted_at")
    df = df.with_columns([
        pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
        pl.col("category").cast(pl.String).cast(pl.Categorical),
    ])
    df = df.drop("skills_text")
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    cmax = df["category"].to_physical().max()
    print(f"[worker] prep done in {time.perf_counter()-t0:.1f}s, "
          f"category codes max={cmax}", flush=True)

    train = df[:n_train]
    val   = df[n_train : n_train + n_val]
    rng = np.random.default_rng(42)
    y_tr = rng.integers(0, 2, size=train.height).astype(np.int8)
    y_v  = rng.integers(0, 2, size=val.height).astype(np.int8)
    print(f"[worker] fitting XGB on n_train={train.height}, n_val={val.height}",
          flush=True)

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


def run_trial(parquet: str, n_train: int, n_val: int,
              timeout: int = 600, log_prefix: str = "") -> str:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
        pickle.dump({"parquet": parquet, "n_train": n_train, "n_val": n_val}, f)
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
    out = (proc.stdout or "") + (proc.stderr or "")
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0 and "FIT_OK" in out:
        outcome = OUTCOME_PASSED
    elif "Traceback" in out:
        outcome = OUTCOME_RAISED
    elif proc.returncode != 0:
        outcome = OUTCOME_CRASHED
    else:
        outcome = OUTCOME_PASSED
    print(f"{log_prefix}n_train={n_train:>8_}, n_val={n_val:>7_} "
          f"-> {outcome} (rc={proc.returncode}) in {elapsed:.1f}s", flush=True)
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--start-train", type=int, default=211_168,
                    help="Upper bound for binary search (must crash).")
    ap.add_argument("--floor",       type=int, default=1_000,
                    help="Lower bound for the search.")
    ap.add_argument("--val-fraction", type=float, default=0.5,
                    help="n_val = max(100, n_train * fraction). Default 0.5.")
    ap.add_argument("--trial-timeout", type=int, default=600)
    args = ap.parse_args()

    def n_val_for(n_train):
        return max(100, int(n_train * args.val_fraction))

    # Confirm baseline crashes.
    base = run_trial(args.parquet, args.start_train, n_val_for(args.start_train),
                     timeout=args.trial_timeout, log_prefix="[base] ")
    if base != OUTCOME_CRASHED:
        print(f"\nBaseline at n_train={args.start_train} did not crash ({base}). "
              f"Cannot bisect. Try a larger --start-train.", flush=True)
        return

    # Try the floor — maybe even tiny crashes (would short-circuit).
    floor = run_trial(args.parquet, args.floor, n_val_for(args.floor),
                      timeout=args.trial_timeout, log_prefix="[floor]")
    if floor == OUTCOME_CRASHED:
        print(f"\nFloor at n_train={args.floor} ALREADY crashes. The trigger "
              f"holds at very small row counts. Reporting floor as the minimum.",
              flush=True)
        print(f"\n=== MIN CRASHING n_train: <= {args.floor} ===", flush=True)
        return

    # Binary search between floor (passes) and start (crashes).
    # Invariant: lo passes, hi crashes. Narrow until hi - lo small.
    lo = args.floor
    hi = args.start_train
    while hi - lo > max(500, hi // 50):
        mid = (lo + hi) // 2
        r = run_trial(args.parquet, mid, n_val_for(mid),
                      timeout=args.trial_timeout, log_prefix="  [mid] ")
        if r == OUTCOME_CRASHED:
            hi = mid
        else:
            lo = mid

    print(f"\n=== MIN CRASHING n_train ~= {hi:_} (lower bound passes at {lo:_}) ===",
          flush=True)
    print(f"  Use this as ROW_LIMIT in the upstream reproducer.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    main()
