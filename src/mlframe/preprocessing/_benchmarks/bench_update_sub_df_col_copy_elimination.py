"""Microbench: ``_update_sub_df_col`` used to do a
full (potentially deep, via copy-on-write triggering on first write) copy of the ENTIRE analysis sub-frame
just to refresh one column's ``value_counts()`` -- called up to 3x per column inside
``analyse_and_clean_features``'s per-column loop whenever ``update_data=True``. The fix reads
``df.loc[analyse_mask, col]`` directly into a fresh Series, never touching/copying ``sub_df``'s other columns.

Run:  CUDA_VISIBLE_DEVICES="" python bench_update_sub_df_col_copy_elimination.py

OLD side = the actual prior implementation (faithfully reproduced from ``git show HEAD``). NEW side = the
shipped helper. Both warmed before timing; best-of-N. Result identity is asserted (same value_counts/nunique).
"""
import sys

sys.modules.setdefault("cupy", None)  # type: ignore[arg-type]  # avoid pre-existing cupy native-AV at import on this box
import time
from gc import collect

import numpy as np
import pandas as pd

from mlframe.preprocessing.cleaning import _update_sub_df_col, analyse_and_clean_features
import mlframe.preprocessing.cleaning as cleaning_module


def best_of(fn, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def _old_update_sub_df_col(df: pd.DataFrame, sub_df: pd.DataFrame, col: str, col_unique_values, nunique, analyse_mask=None):
    """Faithful reproduction of the pre-fix implementation (``git show HEAD:src/mlframe/preprocessing/cleaning.py``)."""
    if analyse_mask is not None:
        sub_df = sub_df.copy() if col in sub_df else sub_df
        sub_df[col] = df.loc[analyse_mask, col]
    col_unique_values = sub_df[col].value_counts(dropna=False)
    nunique = len(col_unique_values)
    collect()
    return col_unique_values, nunique


def make_wide_sparse_frame(n_rows: int, n_cols: int, seed: int = 0) -> pd.DataFrame:
    """A wide frame (many columns) with sparse, rare-value-heavy categorical columns -- the realistic shape
    ``analyse_and_clean_features``'s rare-value-merge loop calls ``_update_sub_df_col`` repeatedly for."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_cols):
        # a handful of common values plus a long tail of rare ones, so most columns trigger a rare-merge call.
        n_rare = 200
        common = rng.integers(0, 4, size=n_rows // 2)
        rare = rng.integers(4, 4 + n_rare, size=n_rows - n_rows // 2)
        vals = np.concatenate([common, rare])
        rng.shuffle(vals)
        data[f"col_{i}"] = vals
    return pd.DataFrame(data)


def bench_isolated_call():
    print("=== _update_sub_df_col: OLD (whole-sub_df copy) vs NEW (direct df.loc[...] value_counts) ===")
    for n_rows, n_cols in ((50_000, 200), (200_000, 200)):
        df = make_wide_sparse_frame(n_rows, n_cols)
        sub_df = df.copy()  # the analysis sub-frame, as analyse_and_clean_features would build it
        analyse_mask = np.ones(n_rows, dtype=bool)
        col = "col_0"

        # col_unique_values / nunique are unconditionally recomputed inside both functions before use,
        # so the placeholder values passed here are never read -- kept type-correct for mypy rather than None.
        _placeholder_counts = pd.Series(dtype="int64")
        old_fn = lambda: _old_update_sub_df_col(df, sub_df, col, _placeholder_counts, 0, analyse_mask=analyse_mask)
        new_fn = lambda: _update_sub_df_col(df, sub_df, col, _placeholder_counts, 0, analyse_mask=analyse_mask)

        old_result, _ = old_fn()
        new_result, _ = new_fn()
        pd.testing.assert_series_equal(old_result.sort_index(), new_result.sort_index())  # nosec B101 - internal identity check, not reachable with untrusted input

        t_old = best_of(old_fn)
        t_new = best_of(new_fn)
        print(f"  n_rows={n_rows:>7} n_cols={n_cols:>4}: OLD {t_old*1e3:8.3f}ms  NEW {t_new*1e3:8.3f}ms  speedup {t_old/t_new:6.2f}x")


def bench_end_to_end():
    print("=== analyse_and_clean_features(update_data=True) end-to-end, wide/sparse frame ===")
    print("    (each call triggers a gc.collect() per refreshed column either way -- that cost is shared by")
    print("     both sides, so it dilutes the copy-elimination delta at the whole-pipeline level; reported")
    print("     honestly, both directions, median-of-7 to damp system noise.)")
    for n_rows, n_cols in ((20_000, 150), (100_000, 150)):
        df = make_wide_sparse_frame(n_rows, n_cols)

        def run_with(update_sub_df_col_fn):
            work = df.copy()
            original = cleaning_module._update_sub_df_col
            cleaning_module._update_sub_df_col = update_sub_df_col_fn
            try:
                analyse_and_clean_features(work, update_data=True, clean_nonnumeric_rarevals=True, verbose=False)
            finally:
                cleaning_module._update_sub_df_col = original

        old_fn = lambda: run_with(_old_update_sub_df_col)
        new_fn = lambda: run_with(_update_sub_df_col)

        # interleaved (old, new, old, new, ...) instead of all-old-then-all-new, so a slow drift in
        # system load over the run doesn't systematically bias one side.
        old_times, new_times = [], []
        for _ in range(7):
            t = time.perf_counter()
            old_fn()
            old_times.append(time.perf_counter() - t)
            t = time.perf_counter()
            new_fn()
            new_times.append(time.perf_counter() - t)
        t_old = float(np.median(old_times))
        t_new = float(np.median(new_times))
        print(f"  n_rows={n_rows:>7} n_cols={n_cols:>4}: OLD median {t_old:7.3f}s  NEW median {t_new:7.3f}s  speedup {t_old/t_new:6.2f}x")
        print(f"    OLD all: {[round(t, 3) for t in old_times]}")
        print(f"    NEW all: {[round(t, 3) for t in new_times]}")


if __name__ == "__main__":
    bench_isolated_call()
    bench_end_to_end()
