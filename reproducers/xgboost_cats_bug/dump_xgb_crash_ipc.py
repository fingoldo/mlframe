r"""Dump the crashing in-memory state to Arrow IPC (not parquet).

Parquet is known NOT to preserve polars Categorical's sparse physical
codes — ``pl.read_parquet`` rebuilds the dict from the values,
yielding compact codes in [0, n_unique-1]. Arrow IPC preserves the
DictionaryArray faithfully including physical code layout.

Hypothesis: an IPC dump of the bisector's crashing train/val frames,
reloaded in a fresh process, will retain sparse codes and therefore
still crash XGB. This would make the IPC bundle the true upstream
reproducer: standalone, self-contained, no polluted cache needed.

Usage:
    python -m mlframe.profiling.dump_xgb_crash_ipc --parquet "R:\..." --out-dir "D:\Temp\xgb_ipc"

Then in a fresh process:
    cd D:\Temp\xgb_ipc
    python reproduce_xgb_crash_ipc.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

DROP_COLS = [
    "uid", "job_posted_at", "job_status", "cl_id",
    "_raw_countries", "_raw_languages", "_raw_tags",
    "job_post_source", "job_post_device", "job_post_flow_type",
]
CULPRIT_CATS = ["category"]
ROW_LIMIT = 211_168


def load_and_prep(parquet_path: str):
    df = pl.read_parquet(parquet_path)
    df = (df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
            .with_columns(pl.col(pl.Utf8).cast(pl.Categorical))
            .sort("job_posted_at"))
    df = df.with_columns([
        pl.col("job_posted_at").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("job_posted_at").dt.day().cast(pl.Int8).alias("day"),
        pl.col("job_posted_at").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("job_posted_at").dt.month().cast(pl.Int8).alias("month"),
    ])
    target = (df["cl_act_total_hired"].fill_null(0) >= 1).cast(pl.Int8).to_numpy()
    df = df.drop([c for c in DROP_COLS if c in df.columns])
    HIGH_CARD = 300
    to_drop_text = [c for c, dt in df.schema.items()
                    if dt in (pl.Categorical, pl.Utf8) and df[c].n_unique() > HIGH_CARD]
    if to_drop_text:
        df = df.drop(to_drop_text)
    all_cat = [c for c, dt in df.schema.items() if dt == pl.Categorical]
    df = df.drop([c for c in all_cat if c not in CULPRIT_CATS])
    for c in CULPRIT_CATS:
        if c in df.columns and df[c].null_count() > 0:
            df = df.with_columns(pl.col(c).fill_null("__MISSING__"))
    n = df.height
    n_test = int(n * 0.10); n_val = int(n * 0.10); n_train = n - n_val - n_test
    train = df[:n_train][:ROW_LIMIT]
    val = df[n_train : n_train + n_val]
    y_tr = target[:n_train][:ROW_LIMIT]
    y_v = target[n_train : n_train + n_val]
    return train, val, y_tr, y_v


REPRO = r'''"""Reproducer for XGB 3.2 access violation on Windows.

Loads Arrow IPC files (preserving polars Categorical's physical
code layout unlike parquet). Fits XGB. Expected: silent kill with
exit code 3221226505 (0xC0000005).

Depends on: polars, xgboost, numpy. No business data (category values
are anonymised to c_NNN tokens in the dump).
"""
import sys, time, numpy as np, polars as pl
from pathlib import Path
from xgboost import XGBClassifier
if sys.platform == "win32":
    import ctypes; ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler; faulthandler.enable(all_threads=True)
except Exception: pass

here = Path(__file__).parent
train = pl.read_ipc(here / "crash_train.arrow")
val   = pl.read_ipc(here / "crash_val.arrow")
y_tr  = np.load(here / "crash_target_train.npy")
y_v   = np.load(here / "crash_target_val.npy")

print(f"train shape={train.shape}, val shape={val.shape}")
for c, dt in train.schema.items():
    if dt == pl.Categorical:
        mn = train[c].to_physical().min()
        mx = train[c].to_physical().max()
        print(f"train[{c}]: n_unique={train[c].n_unique()}, "
              f"physical_codes_range=[{mn}, {mx}], "
              f"distinct={train[c].to_physical().n_unique()}")

print("Calling XGB.fit() — expect silent kill on Windows+xgboost 3.2.0.")
t0 = time.perf_counter()
m = XGBClassifier(
    n_estimators=5, enable_categorical=True, tree_method="hist",
    device="cpu", n_jobs=-1, verbosity=1,
    max_cat_to_onehot=1, max_cat_threshold=100,
    early_stopping_rounds=3,
    objective="binary:logistic", eval_metric="logloss",
)
m.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
print(f"FIT_OK in {time.perf_counter()-t0:.1f}s — bug did NOT reproduce")
'''


def anonymise_categorical(s: pl.Series) -> pl.Series:
    uniq = s.drop_nulls().unique().to_list()
    mapping = {orig: f"c_{i:03d}" for i, orig in enumerate(uniq)}
    return s.cast(pl.String).replace_strict(
        list(mapping.keys()), list(mapping.values()), default=None,
    ).cast(pl.Categorical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--anonymise", action="store_true", default=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tr, va, y_tr, y_v = load_and_prep(args.parquet)
    print(f"prepared in {time.perf_counter()-t0:.1f}s, "
          f"train={tr.shape}, val={va.shape}", flush=True)

    for c in CULPRIT_CATS:
        if c in tr.columns:
            mn = tr[c].to_physical().min()
            mx = tr[c].to_physical().max()
            print(f"before dump: train[{c}] physical_codes_range=[{mn}, {mx}], "
                  f"distinct={tr[c].to_physical().n_unique()}", flush=True)

    if args.anonymise:
        print("Anonymising culprit column values to c_NNN...", flush=True)
        for c in CULPRIT_CATS:
            if c in tr.columns:
                tr = tr.with_columns(anonymise_categorical(tr[c]).alias(c))
                va = va.with_columns(anonymise_categorical(va[c]).alias(c))
        for c in CULPRIT_CATS:
            if c in tr.columns:
                mn = tr[c].to_physical().min()
                mx = tr[c].to_physical().max()
                print(f"after anonymise: train[{c}] physical_codes_range=[{mn}, {mx}], "
                      f"distinct={tr[c].to_physical().n_unique()}", flush=True)

    print("Writing Arrow IPC dumps...", flush=True)
    tr.write_ipc(out / "crash_train.arrow")
    va.write_ipc(out / "crash_val.arrow")
    np.save(out / "crash_target_train.npy", y_tr)
    np.save(out / "crash_target_val.npy", y_v)
    (out / "reproduce_xgb_crash_ipc.py").write_text(REPRO, encoding="utf-8")

    total = 0
    for f in out.iterdir():
        sz = f.stat().st_size; total += sz
        print(f"  {f.name}: {sz/1e6:.2f} MB", flush=True)
    print(f"total bundle: {total/1e6:.2f} MB", flush=True)
    print(f"\nVerify:\n  cd {out}\n  python reproduce_xgb_crash_ipc.py", flush=True)


if __name__ == "__main__":
    main()
