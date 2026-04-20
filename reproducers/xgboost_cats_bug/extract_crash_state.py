r"""Extract the EXACT polars/Arrow state of the 2 fit rows that
trigger the silent kill, so a synthetic reproducer can recreate
them position-by-position without needing the full 2.5M-row parquet.

Runs the same minimal pipeline as ``no_replace_check.py``, but
right before the crash-causing ``XGBClassifier.fit()`` call it
prints:
  - The category dict length
  - The 2 physical codes (train row, val row)
  - The 2 string values backing those codes
  - The full set of unique physical codes in the entire frame
    (not just the 2 fit rows — these are needed to know what cache
    positions the synthetic must reach)

Usage:
    python -m mlframe.profiling.extract_crash_state \
        --parquet "R:\\..\\jobs_details.parquet"

Output is a Python literal block ready to paste into the synthetic
reproducer.
"""
from __future__ import annotations

import argparse
import sys
import time

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()

    print(f"polars {pl.__version__}, using_string_cache={pl.using_string_cache()}",
          flush=True)

    t0 = time.perf_counter()
    df = pl.read_parquet(args.parquet,
                         columns=["skills_text", "category", "job_posted_at"])
    df = df.sort("job_posted_at").drop("job_posted_at")
    df = df.with_columns([
        pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
        pl.col("category").cast(pl.String).cast(pl.Categorical),
    ])
    df = df.drop("skills_text")
    df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
    print(f"prep done in {time.perf_counter()-t0:.1f}s", flush=True)

    cat = df["category"]
    print(f"\n=== State just before crash-causing XGB.fit() ===", flush=True)
    print(f"  category dtype: {cat.dtype}", flush=True)
    print(f"  category n_unique: {cat.n_unique()}", flush=True)
    print(f"  category n_chunks: {cat.n_chunks()}", flush=True)

    phys = cat.to_physical()
    print(f"  physical codes range: [{phys.min()}, {phys.max()}]", flush=True)
    print(f"  distinct physical codes used: {phys.n_unique()}", flush=True)

    # The 2 fit rows.
    train_row_code = int(phys[0])
    val_row_code   = int(phys[1])
    train_row_str  = cat[0]
    val_row_str    = cat[1]
    print(f"\n=== The 2 fit rows ===", flush=True)
    print(f"  train[0]: physical_code={train_row_code}, string={train_row_str!r}",
          flush=True)
    print(f"  val  [0]: physical_code={val_row_code}, string={val_row_str!r}",
          flush=True)

    # Get full list of (string, physical_code) for all unique values
    # in category. This is the exact dict slice we need to recreate.
    # n_unique should be small (~89-90 in prod).
    print(f"\n=== Full (string -> physical_code) mapping for category ===",
          flush=True)
    # Sort by physical code for readability.
    pairs = []
    seen = set()
    for i in range(cat.len()):
        c = int(phys[i])
        if c in seen:
            continue
        seen.add(c)
        s = cat[i]
        pairs.append((c, s))
        if len(seen) >= cat.n_unique():
            break
    pairs.sort()
    for c, s in pairs:
        print(f"  code={c:>10}  ->  {s!r}", flush=True)

    # Emit Python literal ready for synthetic reproducer.
    print(f"\n=== PASTE INTO SYNTHETIC REPRODUCER ===", flush=True)
    print(f"# Each tuple is (physical_code, string_value).", flush=True)
    print(f"# To reproduce, the synthetic must populate the polars global", flush=True)
    print(f"# StringCache so each string ends up at its physical_code.", flush=True)
    print(f"DICT_PAIRS = [", flush=True)
    for c, s in pairs:
        print(f"    ({c}, {s!r}),", flush=True)
    print(f"]", flush=True)
    print(f"TRAIN_CODE = {train_row_code}  # row[0] physical code", flush=True)
    print(f"VAL_CODE   = {val_row_code}  # row[1] physical code", flush=True)
    print(f"DICT_TOTAL_SIZE_HINT = {phys.max() + 1}  # max code + 1; pad cache up to this",
          flush=True)


if __name__ == "__main__":
    main()
