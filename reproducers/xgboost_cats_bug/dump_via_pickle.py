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
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import polars as pl


def _spawn_fit_check(loader_code: str, log_prefix: str) -> tuple[str, int]:
    """Run loader_code in a fresh subprocess; loader_code must define
    `train`, `val`, `y_tr`, `y_v` then call XGB.fit. Returns
    (outcome, exit_code).
    """
    script = textwrap.dedent("""
        import sys, time, numpy as np, polars as pl
        from xgboost import XGBClassifier
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
        try:
            import faulthandler; faulthandler.enable(all_threads=True)
        except Exception:
            pass

        # --- format-specific loader ---
    """) + loader_code + textwrap.dedent("""

        print(f"loaded train={train.shape}, val={val.shape}, "
              f"category codes train={train['category'].to_physical().to_list()}, "
              f"val={val['category'].to_physical().to_list()}", flush=True)

        m = XGBClassifier(
            n_estimators=5, enable_categorical=True, tree_method="hist",
            device="cpu", n_jobs=-1, verbosity=1,
            max_cat_to_onehot=1, max_cat_threshold=100,
            early_stopping_rounds=3,
            objective="binary:logistic", eval_metric="logloss",
        )
        try:
            m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
            print("FIT_OK")
        except BaseException as e:
            print(f"RAISED {type(e).__name__}: {e}")
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w",
                                     encoding="utf-8") as f:
        f.write(script)
        sf = f.name
    try:
        proc = subprocess.run([sys.executable, sf], capture_output=True,
                              text=True, timeout=300, encoding="utf-8",
                              errors="replace")
    finally:
        os.unlink(sf)
    out = (proc.stdout or "") + (proc.stderr or "")
    rc = proc.returncode
    if "FIT_OK" in out:
        outcome = "passed"
    elif "RAISED" in out:
        outcome = "raised"
    elif rc != 0:
        outcome = "crashed"
    else:
        outcome = "passed"
    print(f"{log_prefix}fit -> {outcome} (rc={rc})", flush=True)
    if outcome == "raised":
        # show the raised line
        for line in out.splitlines():
            if "RAISED" in line:
                print(f"{log_prefix}  {line.strip()}", flush=True)
                break
    return outcome, rc


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

    orig_codes = train["category"].to_physical().to_list()
    orig_max = df["category"].to_physical().max()

    def crash_check(format_name: str, loader_code: str, prefix: str):
        outcome, rc = _spawn_fit_check(loader_code, prefix)
        return outcome, rc

    # --- pickle ---
    print("\n--- pickle ---", flush=True)
    pkl = out / "train_val.pkl"
    with open(pkl, "wb") as f:
        pickle.dump((train, val), f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = pkl.stat().st_size
    print(f"  saved {sz/1024:.1f} KB", flush=True)
    with open(pkl, "rb") as f:
        tr2, v2 = pickle.load(f)
    print(f"  reloaded train phys: {tr2['category'].to_physical().to_list()}",
          flush=True)
    if tr2["category"].to_physical().to_list() == orig_codes:
        print(f"  >>> pickle PRESERVED codes <<<", flush=True)
    crash_check("pickle", textwrap.dedent(f"""
        import pickle, numpy as np
        with open(r'{pkl}', 'rb') as f:
            train, val = pickle.load(f)
        y_tr = np.array([0], dtype=np.int8)
        y_v  = np.array([1], dtype=np.int8)
    """), "  [pickle] ")

    # --- IPC train+val ---
    print("\n--- IPC train+val ---", flush=True)
    ipc_t = out / "train.arrow"
    ipc_v = out / "val.arrow"
    train.write_ipc(ipc_t)
    val.write_ipc(ipc_v)
    sz = ipc_t.stat().st_size + ipc_v.stat().st_size
    print(f"  saved {sz/1024:.1f} KB", flush=True)
    tr3 = pl.read_ipc(ipc_t)
    print(f"  reloaded train phys: {tr3['category'].to_physical().to_list()}",
          flush=True)
    if tr3["category"].to_physical().to_list() == orig_codes:
        print(f"  >>> IPC train+val PRESERVED codes <<<", flush=True)
    crash_check("ipc_train_val", textwrap.dedent(f"""
        import polars as pl, numpy as np
        train = pl.read_ipc(r'{ipc_t}')
        val   = pl.read_ipc(r'{ipc_v}')
        y_tr = np.array([0], dtype=np.int8)
        y_v  = np.array([1], dtype=np.int8)
    """), "  [ipc_tv] ")

    # --- IPC full df, sliced after load ---
    print("\n--- IPC full df ---", flush=True)
    ipc_full = out / "full_df.arrow"
    df.write_ipc(ipc_full)
    sz = ipc_full.stat().st_size
    print(f"  saved {sz/1e6:.2f} MB (unsliced source)", flush=True)
    df2 = pl.read_ipc(ipc_full)
    print(f"  reloaded full df phys max: {df2['category'].to_physical().max()}",
          flush=True)
    if df2["category"].to_physical().max() == orig_max:
        print(f"  >>> IPC full df PRESERVED codes <<<", flush=True)
    crash_check("ipc_full", textwrap.dedent(f"""
        import polars as pl, numpy as np
        df = pl.read_ipc(r'{ipc_full}')
        train = df[:1]
        val   = df[1:2]
        y_tr = np.array([0], dtype=np.int8)
        y_v  = np.array([1], dtype=np.int8)
    """), "  [ipc_full] ")

    # --- pickle full df, slice after load ---
    print("\n--- pickle full df ---", flush=True)
    pkl_full = out / "full_df.pkl"
    with open(pkl_full, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = pkl_full.stat().st_size
    print(f"  saved {sz/1e6:.2f} MB", flush=True)
    crash_check("pickle_full", textwrap.dedent(f"""
        import pickle, numpy as np
        with open(r'{pkl_full}', 'rb') as f:
            df = pickle.load(f)
        train = df[:1]
        val   = df[1:2]
        y_tr = np.array([0], dtype=np.int8)
        y_v  = np.array([1], dtype=np.int8)
    """), "  [pkl_full] ")

    print("\n=== Summary ===", flush=True)
    print("  Look for the smallest format whose 'fit -> crashed' line appears.",
          flush=True)
    print("  That format's file is the upstream-shippable bundle.", flush=True)


if __name__ == "__main__":
    main()
