r"""Bisect which VALUES of skills_text are needed to trigger the XGB
silent kill. Binary search on the set of unique string values.

At each step:
  1. Take a random half of the current candidate set of unique
     values.
  2. In the real parquet column, replace every value NOT in the half
     with a single sentinel ``__DROPPED__``. This preserves row
     count, category null pattern, and fills_text's scatter
     structure for the retained values.
  3. Run the minimal pipeline (cast both skills_text + category,
     drop skills_text, fill_null on category, XGB fit) in a
     subprocess.
  4. If it crashes (rc=0xC0000005), recurse into that half. If it
     passes, the culprit is in the other half.

Prints max string length of the retained subset at each step, in
case crash correlates with specific-length bytes.

Usage:
    python -m mlframe.profiling.bisect_skills_text_values \
        --parquet "R:\\..\\jobs_details.parquet"

Runtime estimate on 128 GB prod box, bisecting from ~2M values:
  log2(2_000_000) ≈ 21 levels. Each trial ~30-60s
  -> ~15-30 min total.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    except Exception:
        pass


DROPPED_SENTINEL = "__DROPPED__"
N_TRAIN = 211_168
N_VAL = 100_000

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
    keep_values = set(state["keep_values"])

    # Load skills_text + category + timestamp (for sort). Target is
    # random here because XGB just needs something to train on.
    t0 = time.perf_counter()
    df = pl.read_parquet(parquet, columns=["skills_text", "category", "job_posted_at"])
    print(f"[worker] loaded {df.shape} in {time.perf_counter()-t0:.1f}s",
          flush=True)

    # Replace values NOT in keep_values with sentinel. Uses replace_strict
    # via a mapping dict to avoid when/then which may rebuild the column
    # in a state-normalizing way.
    t0 = time.perf_counter()
    if keep_values:
        df = df.with_columns(
            pl.when(pl.col("skills_text").is_in(list(keep_values)))
              .then(pl.col("skills_text"))
              .otherwise(pl.lit(DROPPED_SENTINEL))
              .alias("skills_text")
        )
    else:
        df = df.with_columns(pl.lit(DROPPED_SENTINEL).alias("skills_text"))

    # Sort by timestamp — mimics minimizer's prod pipeline step.
    df = df.sort("job_posted_at")
    df = df.drop("job_posted_at")

    # Force skills_text + category through shared-cache batch.
    df = df.with_columns([
        pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
        pl.col("category").cast(pl.String).cast(pl.Categorical),
    ])
    print(f"[worker] cast done in {time.perf_counter()-t0:.1f}s, "
          f"skills_text n_unique={df['skills_text'].n_unique()}, "
          f"category n_unique={df['category'].n_unique()}",
          flush=True)

    # Drop skills_text, fill_null on category.
    df = df.drop("skills_text")
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    codes_max = df["category"].to_physical().max()
    print(f"[worker] category physical_codes_max after fill_null: {codes_max}",
          flush=True)

    # Slice.
    train = df[:N_TRAIN]
    val   = df[N_TRAIN : N_TRAIN + N_VAL]
    rng = np.random.default_rng(42)
    y_tr = rng.integers(0, 2, size=train.height).astype(np.int8)
    y_v  = rng.integers(0, 2, size=val.height).astype(np.int8)

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


def run_trial(parquet: str, keep_values: Sequence[str],
              timeout: int = 600, log_prefix: str = "") -> str:
    max_len = max((len(v) for v in keep_values), default=0)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
        pickle.dump({
            "parquet": parquet,
            "keep_values": list(keep_values),
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
    print(f"{log_prefix}trial keep={len(keep_values):_}, "
          f"max_len={max_len} -> {outcome} (rc={proc.returncode}) "
          f"in {elapsed:.1f}s", flush=True)
    return outcome


def enumerate_all_skills(parquet_path: str) -> List[str]:
    import polars as pl
    print(f"Enumerating skills_text uniques...", flush=True)
    t0 = time.perf_counter()
    df = pl.read_parquet(parquet_path, columns=["skills_text"])
    vals = df["skills_text"].drop_nulls().unique().to_list()
    print(f"  found {len(vals):_} unique values in "
          f"{time.perf_counter()-t0:.1f}s, max_len={max(len(v) for v in vals)}",
          flush=True)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--trial-timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the random half split at each level.")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    all_values = enumerate_all_skills(args.parquet)

    # Baseline
    base = run_trial(args.parquet, all_values, timeout=args.trial_timeout,
                     log_prefix="[base]   ")
    if base != OUTCOME_CRASHED:
        print(f"Baseline did not crash ({base}) — cannot bisect. Abort.",
              flush=True)
        return

    # Zero-shortcut: maybe even an empty keep_values set crashes
    # (meaning skills_text can be all-sentinel — the very act of casting
    # one cache-polluting column is enough).
    empty = run_trial(args.parquet, [], timeout=args.trial_timeout,
                      log_prefix="[empty]  ")
    if empty == OUTCOME_CRASHED:
        print(f"  ZERO skills values retained and still crashed — "
              f"the sentinel alone is enough. Skipping bisection.",
              flush=True)
        return

    current = list(all_values)
    level = 0
    while len(current) > 1:
        level += 1
        rng.shuffle(current)
        mid = len(current) // 2
        left = current[:mid]
        right = current[mid:]

        lr = run_trial(args.parquet, left, timeout=args.trial_timeout,
                       log_prefix=f"  [L{level:02d}] ")
        if lr == OUTCOME_CRASHED:
            current = left
            continue
        rr = run_trial(args.parquet, right, timeout=args.trial_timeout,
                       log_prefix=f"  [R{level:02d}] ")
        if rr == OUTCOME_CRASHED:
            current = right
            continue
        # Neither half crashes alone — interaction required.
        print(f"  L{level:02d}: interaction (neither half alone); "
              f"keeping current {len(current):_} values as minimum.",
              flush=True)
        break

    print(f"\n=== MINIMUM VALUES THAT STILL CRASH: {len(current):_} ===",
          flush=True)
    if len(current) <= 50:
        print(f"  sample: {current[:20]}", flush=True)
    print(f"  max_len: {max(len(v) for v in current) if current else 0}",
          flush=True)
    print(f"  min_len: {min(len(v) for v in current) if current else 0}",
          flush=True)
    # Save the minimal set for convenience.
    out = Path("bisect_skills_text_values_minimum.txt").absolute()
    out.write_text("\n".join(current), encoding="utf-8")
    print(f"  saved to {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2]))
    main()
