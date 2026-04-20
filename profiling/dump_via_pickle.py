r"""Try saving the 2-row crash-state polars DataFrame via pickle, IPC,
and feather formats. See which (if any) preserves the sparse
Categorical physical codes on reload.

If pickle preserves the state, the bundle for the upstream issue is
just a tiny pickle file (tens of KB), not a 50-100MB parquet.

Usage:
    python -m mlframe.profiling.dump_via_pickle \
        --parquet "R:\\..\\jobs_details.parquet" \
        --out-dir "D:\\Temp\\xgb_pickle_bundle"

Then:
    cd D:\\Temp\\xgb_pickle_bundle
    python reload_and_fit.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    print(f"polars {pl.__version__}, using_string_cache={pl.using_string_cache()}",
          flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Reproduce the crash state from prod.
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

    train = df[:1]
    val   = df[1:2]
    print(f"train phys: {train['category'].to_physical().to_list()}", flush=True)
    print(f"val   phys: {val['category'].to_physical().to_list()}", flush=True)
    print(f"category n_unique={train['category'].n_unique()}, "
          f"max_code in full df={df['category'].to_physical().max()}", flush=True)

    # Try pickle.
    print("\n--- pickle ---", flush=True)
    pkl = out / "train_val.pkl"
    with open(pkl, "wb") as f:
        pickle.dump((train, val), f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = pkl.stat().st_size
    print(f"  saved {sz/1024:.1f} KB", flush=True)
    # Reload & inspect codes.
    with open(pkl, "rb") as f:
        tr2, v2 = pickle.load(f)
    print(f"  reloaded train phys: {tr2['category'].to_physical().to_list()}",
          flush=True)
    print(f"  reloaded val   phys: {v2['category'].to_physical().to_list()}",
          flush=True)
    if tr2["category"].to_physical().to_list() == train["category"].to_physical().to_list():
        print(f"  >>> pickle PRESERVED codes <<<", flush=True)

    # Try IPC.
    print("\n--- IPC ---", flush=True)
    ipc_t = out / "train.arrow"
    ipc_v = out / "val.arrow"
    train.write_ipc(ipc_t)
    val.write_ipc(ipc_v)
    sz = ipc_t.stat().st_size + ipc_v.stat().st_size
    print(f"  saved {sz/1024:.1f} KB", flush=True)
    tr3 = pl.read_ipc(ipc_t)
    v3  = pl.read_ipc(ipc_v)
    print(f"  reloaded train phys: {tr3['category'].to_physical().to_list()}",
          flush=True)
    if tr3["category"].to_physical().to_list() == train["category"].to_physical().to_list():
        print(f"  >>> IPC PRESERVED codes <<<", flush=True)

    # Try writing the FULL df via IPC (so the dict is preserved in the source frame).
    print("\n--- IPC full df ---", flush=True)
    ipc_full = out / "full_df.arrow"
    df.write_ipc(ipc_full)
    sz = ipc_full.stat().st_size
    print(f"  saved {sz/1e6:.2f} MB (this is the unsliced source)", flush=True)
    df2 = pl.read_ipc(ipc_full)
    print(f"  reloaded full df phys max: {df2['category'].to_physical().max()}",
          flush=True)
    if df2["category"].to_physical().max() == df["category"].to_physical().max():
        print(f"  >>> IPC PRESERVED codes on full df <<<", flush=True)


if __name__ == "__main__":
    main()
