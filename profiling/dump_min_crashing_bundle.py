r"""Dump the minimum-row parquet that still triggers the silent kill.

Trigger requirements (confirmed by row-bisection 2026-04-20):
  - skills_text column has ~2.5M unique values, ALL needed in cache
  - category column has ~89 unique values + nulls
  - even n_train=1 crashes — train size irrelevant
  - val DMatrix construction is what dies; val_size=100 sufficient

So minimum parquet = 2.5M rows where each row carries one unique
skills_text value, and category cycles through its real values.
job_posted_at provides sortable timestamps.

The script:
  1. Loads the production parquet
  2. Drops duplicates of skills_text (keep first occurrence) -> ~2.5M rows
  3. Replaces category in those kept rows with random selection from
     the original category dict (preserves cardinality + nulls)
  4. Anonymises skills_text to opaque tokens preserving length
  5. Writes the small bundle parquet + an upstream-ready reproducer

Usage:
    python -m mlframe.profiling.dump_min_crashing_bundle \
        --parquet "R:\\..\\jobs_details.parquet" \
        --out-dir "D:\\Temp\\xgb_min_bundle"

Then on the same machine:
    cd D:\\Temp\\xgb_min_bundle
    python reproduce.py

Expected: silent kill 0xC0000005.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


REPRO_SCRIPT = r'''"""Minimal XGBoost 3.2 access violation reproducer.

Bundle contents (must be in the same dir as this script):
  bundle.parquet  — ~2.5M rows x 3 cols (anonymised skills_text,
                    category, job_posted_at)

Expected on Windows + xgboost 3.2.0 + polars 1.33+:
  Silent process exit, code 3221226505 (0xC0000005, STATUS_ACCESS_VIOLATION).
  No Python traceback. No C++ exception. fault dies between the
  first FIT_OK-bound print and the actual fit.

Workaround: cast category to pl.Enum(sorted_uniques) after fill_null.
"""
import sys, time, numpy as np, polars as pl
from pathlib import Path
from xgboost import XGBClassifier

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception:
    pass

here = Path(__file__).parent
print(f"polars {pl.__version__}, using_string_cache={pl.using_string_cache()}")

t0 = time.perf_counter()
df = pl.read_parquet(here / "bundle.parquet")
print(f"loaded {df.shape} in {time.perf_counter()-t0:.1f}s")

df = df.sort("job_posted_at").drop("job_posted_at")
df = df.with_columns([
    pl.col("skills_text").cast(pl.String).cast(pl.Categorical),
    pl.col("category").cast(pl.String).cast(pl.Categorical),
])
print(f"after cast: skills n_unique={df['skills_text'].n_unique():_}, "
      f"category n_unique={df['category'].n_unique()}")
print(f"  category codes pre-fill_null: max="
      f"{df['category'].to_physical().drop_nulls().max()}")

df = df.drop("skills_text")
df = df.with_columns(pl.col("category").fill_null("__MISSING__"))
print(f"  category codes post-fill_null: max="
      f"{df['category'].to_physical().max()}, "
      f"distinct={df['category'].to_physical().n_unique()}")

train = df[:1]
val = df[1:101]
y_tr = np.array([0], dtype=np.int8)
y_v = np.random.default_rng(42).integers(0, 2, 100).astype(np.int8)
print(f"train={train.shape}, val={val.shape}")
print("fitting XGB — expect silent kill exit 3221226505")

m = XGBClassifier(
    n_estimators=5, enable_categorical=True, tree_method="hist",
    device="cpu", n_jobs=-1, verbosity=1,
    max_cat_to_onehot=1, max_cat_threshold=100,
    early_stopping_rounds=3,
    objective="binary:logistic", eval_metric="logloss",
)
t0 = time.perf_counter()
m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
print(f"FIT_OK in {time.perf_counter()-t0:.1f}s — bug did NOT reproduce!")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loading source parquet...", flush=True)
    t0 = time.perf_counter()
    df = pl.read_parquet(args.parquet, columns=["skills_text", "category", "job_posted_at"])
    print(f"  loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Keep one row per unique skills_text value (preserves cache pollution).
    print("deduplicating skills_text (one row per unique value)...", flush=True)
    t0 = time.perf_counter()
    dedup = df.unique(subset=["skills_text"], keep="first", maintain_order=True)
    print(f"  shape after dedup: {dedup.shape} ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    # Anonymise skills_text — replace each unique value with c{i:08d}.
    # Preserves cardinality and length-distribution similarity.
    print("anonymising skills_text...", flush=True)
    t0 = time.perf_counter()
    n = dedup.height
    anon = [f"s{i:08d}" for i in range(n)]
    dedup = dedup.with_columns(pl.Series("skills_text", anon, dtype=pl.String))
    print(f"  done ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Write bundle.
    bundle_path = out / "bundle.parquet"
    print(f"writing bundle to {bundle_path}...", flush=True)
    t0 = time.perf_counter()
    dedup.write_parquet(bundle_path, compression="zstd")
    sz = bundle_path.stat().st_size / 1e6
    print(f"  done in {time.perf_counter()-t0:.1f}s, size={sz:.2f}MB", flush=True)

    # Write reproducer.
    repro_path = out / "reproduce.py"
    repro_path.write_text(REPRO_SCRIPT, encoding="utf-8")
    print(f"wrote reproducer: {repro_path}", flush=True)

    print(f"\nVerify the bundle reproduces the crash:", flush=True)
    print(f"  cd {out}", flush=True)
    print(f"  python reproduce.py", flush=True)
    print(f"\nExpected: 'fitting XGB' line printed, then silent exit "
          f"3221226505 (no FIT_OK).", flush=True)


if __name__ == "__main__":
    main()
