r"""Bisect: how few SOURCE-PARQUET rows still trigger the silent kill?

We've shown: minimum train+val for fit = 1+1. Now we vary how many
rows of the source parquet to load. Smaller subset = less cache
pollution from skills_text = potentially smaller bundle.

Trial:
  1. Load skills_text + category + job_posted_at, slice [:N].
  2. Same minimal pipeline as no_replace_check.py (sort, cast, drop,
     fill_null).
  3. Slice train=[:1], val=[1:2] from the post-pipeline frame.
  4. Fit XGB. Outcome: crashed | passed | raised.

Subprocess-isolated. Binary-search N from 9_018_479 down to 100.

Usage:
    python -m mlframe.profiling.bisect_source_rows \
        --parquet "R:\\..\\jobs_details.parquet"
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
    n_rows  = state["n_rows"]

    t0 = time.perf_counter()
    df = pl.read_parquet(parquet, columns=["skills_text", "category", "job_posted_at"],
                          n_rows=n_rows)
    df = df.sort("job_posted_at").drop("job_posted_at")
    df = df.with_columns([
        pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
        pl.col("category").cast(pl.String).cast(pl.Categorical),
    ])
    df = df.drop("skills_text")
    if df["category"].null_count() > 0:
        df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    cmax = df["category"].to_physical().max()
    nu = df["category"].n_unique()
    print(f"[worker] n_rows={df.height}, category n_unique={nu}, "
          f"max_code={cmax} (prep {time.perf_counter()-t0:.1f}s)", flush=True)

    if df.height < 2:
        print(f"[worker] need >=2 rows for train+val", flush=True)
        return 5

    train = df[:1]
    val   = df[1:2]
    rng = np.random.default_rng(42)
    y_tr = rng.integers(0, 2, train.height).astype(np.int8)
    y_v  = rng.integers(0, 2, val.height).astype(np.int8)

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


def run_trial(parquet: str, n_rows: int, timeout: int = 600,
              log_prefix: str = "") -> str:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
        pickle.dump({"parquet": parquet, "n_rows": n_rows}, f)
        sf = f.name
    cmd = [sys.executable, __file__, "--worker", sf]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        os.unlink(sf)
        return OUTCOME_TIMEOUT
    os.unlink(sf)
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
    # Extract n_unique + max_code from worker output for logging.
    extra = ""
    for line in out.splitlines():
        if "[worker]" in line and "max_code" in line:
            extra = line.replace("[worker]", "").strip()
            break
    print(f"{log_prefix}n_rows={n_rows:>10_} -> {outcome} (rc={proc.returncode}) "
          f"{elapsed:.1f}s | {extra}", flush=True)
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--start", type=int, default=9_018_479)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--trial-timeout", type=int, default=600)
    args = ap.parse_args()

    base = run_trial(args.parquet, args.start, timeout=args.trial_timeout,
                     log_prefix="[base]  ")
    if base != OUTCOME_CRASHED:
        print(f"\nbaseline at {args.start} did not crash ({base}). Abort.",
              flush=True)
        return

    fl = run_trial(args.parquet, args.floor, timeout=args.trial_timeout,
                   log_prefix="[floor] ")
    if fl == OUTCOME_CRASHED:
        print(f"\nfloor at {args.floor} ALREADY crashes — minimum <= {args.floor}",
              flush=True)
        print(f"\n=== MIN CRASHING SOURCE ROWS: <= {args.floor} ===", flush=True)
        return

    lo, hi = args.floor, args.start
    while hi - lo > max(100, hi // 50):
        mid = (lo + hi) // 2
        r = run_trial(args.parquet, mid, timeout=args.trial_timeout,
                      log_prefix="  [mid] ")
        if r == OUTCOME_CRASHED:
            hi = mid
        else:
            lo = mid

    print(f"\n=== MIN CRASHING SOURCE ROWS ~= {hi:_} (passes at {lo:_}) ===",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    main()
