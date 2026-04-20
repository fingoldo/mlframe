r"""Targeted test: does the existing bundle crash when loaded with a
polluted polars StringCache?

Background
----------
The bundle at ``D:\Temp\xgb_crash_slice\`` (``crash_train.parquet`` et al.)
was produced by ``dump_xgb_prodcrash_slice.py``. When its standalone
``reproduce_xgb_crash.py`` loads the parquet and fits XGB, the fit
completes cleanly (``FIT_OK``) because parquet-read in a fresh process
rebuilds the pl.Categorical dict with compact physical codes
(``[0, 48]``).

The bisector's worker — which crashes — goes through the same
preprocessing but *in the same process* as the earlier casts, so
polars' global StringCache carries over the sparse physical codes
from other categorical columns that were cast earlier (~3.3M total
distinct strings across ``skills_text``, ``ontology_skills_text``,
``_raw_segmentation``, ``_raw_countries``, etc.). ``category``'s 49
values end up at scattered physical codes in ``[2, 3287945]``.

This script polutes the StringCache first, THEN loads the bundle's
parquets. If the resulting frames have the same prod layout but
sparse physical codes in ``category``, and fit crashes, the trigger
is specifically ``sparse physical codes on pl.Categorical`` in a
real-data context (dtype mix, null pattern, etc.) — a narrower
finding than "sparse codes alone crash XGB".

Usage
-----
    python -m mlframe.profiling.reproduce_xgb_bundle_with_sparse_codes \
        --bundle-dir D:\\Temp\\xgb_crash_slice \
        [--padding-uniques 3_500_000]  # default ~= user prod cache size

Expected on Windows + xgboost 3.2.0:
    exit 3221226505 (0xC0000005 access violation), no traceback.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def pollute_string_cache(padding_uniques: int) -> None:
    """Cast a huge pool of unique padding strings to pl.Categorical to
    populate the polars global StringCache so that any subsequent
    pl.Categorical casts in this process receive high physical codes."""
    print(f"Polluting StringCache with {padding_uniques:_} unique padding strings...",
          flush=True)
    t0 = time.perf_counter()
    padding = [f"__pad_{i:08d}" for i in range(padding_uniques)]
    _ = pl.Series("_pollute", padding, dtype=pl.Categorical)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True, type=str)
    ap.add_argument("--padding-uniques", type=int, default=3_500_000,
                    help="Approximate number of unique strings to pollute the "
                         "StringCache with before loading the bundle. Default 3.5M "
                         "matches the scale of the user's real prod StringCache.")
    args = ap.parse_args()

    from xgboost import XGBClassifier

    bundle = Path(args.bundle_dir)

    # STEP 1: pollute the StringCache BEFORE loading the bundle.
    pollute_string_cache(args.padding_uniques)

    # STEP 2: load bundle parquets. pl.Categorical columns inside will
    # receive physical codes from the now-polluted cache.
    print(f"Loading bundle from {bundle}...", flush=True)
    train = pl.read_parquet(bundle / "crash_train.parquet")
    val   = pl.read_parquet(bundle / "crash_val.parquet")
    y_tr  = np.load(bundle / "crash_target_train.npy")
    y_v   = np.load(bundle / "crash_target_val.npy")
    print(f"  train shape={train.shape}, val shape={val.shape}", flush=True)

    # Inspect the `category` column's physical code range.
    for c in [c for c, dt in train.schema.items() if dt == pl.Categorical]:
        mn = train[c].to_physical().min()
        mx = train[c].to_physical().max()
        nu = train[c].n_unique()
        dc = train[c].to_physical().n_unique()
        print(f"  train[{c}]: n_unique={nu}, "
              f"physical_codes_range=[{mn}, {mx}], distinct_codes={dc}",
              flush=True)

    print("Calling XGBClassifier.fit() — expect silent kill (exit 3221226505) "
          "if sparse codes on bundle data is the trigger.", flush=True)
    t0 = time.perf_counter()
    model = XGBClassifier(
        n_estimators=5, enable_categorical=True, tree_method="hist",
        device="cpu", n_jobs=-1, verbosity=1,
        max_cat_to_onehot=1, max_cat_threshold=100,
        early_stopping_rounds=3,
        objective="binary:logistic", eval_metric="logloss",
    )
    try:
        model.fit(train, y_tr, eval_set=[(val, y_v)], verbose=False)
        print(f"FIT_OK in {time.perf_counter()-t0:.1f}s — sparse codes on "
              f"bundle did NOT crash. Either: (a) bundle's parquet-read "
              f"assigned compact codes anyway (cache gets rebuilt on read?), "
              f"or (b) sparse codes are not the trigger even with prod data.",
              flush=True)
    except BaseException as e:
        print(f"RAISED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
