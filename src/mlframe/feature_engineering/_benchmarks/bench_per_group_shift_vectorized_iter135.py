"""A/B bench: per_group_shift Python-segment-loop (OLD) vs fully-vectorized (NEW).

OLD: iterate every group in Python, doing one fancy-index read + one scatter write
per group. At many small groups (200k groups / ~50 rows) the Python loop dominates.

NEW: build the within-group rank of each sorted row once (vectorized via segment
lengths), gather source sorted positions p-n, mask out rows whose source falls in a
PRIOR group, scatter all at once. Zero Python per-group iteration.

Run: CUDA_VISIBLE_DEVICES="" python bench_per_group_shift_vectorized_iter135.py
"""
import sys
sys.modules["cupy"] = None
import time

import numpy as np

from mlframe.feature_engineering.grouped import iter_group_segments, per_group_shift


def _old_per_group_shift(values, group_ids, n=1, *, fill_value=np.nan, output_dtype=np.float64):
    values_arr = np.ascontiguousarray(values)
    out = np.full(values_arr.size, fill_value, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        seg_len = seg_idx.size
        if n > 0:
            if seg_len <= n:
                continue
            out[seg_idx[n:]] = values_arr[seg_idx[:-n]]
        elif n < 0:
            k = -n
            if seg_len <= k:
                continue
            out[seg_idx[:-k]] = values_arr[seg_idx[k:]]
        else:
            out[seg_idx] = values_arr[seg_idx]
    return out


def _bestof(fn, args, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for n, n_groups in [(1_000_000, 20_000), (10_000_000, 200_000), (1_000_000, 5)]:
        gids = rng.integers(0, n_groups, size=n).astype(np.int64)
        vals = rng.standard_normal(n)
        for shift in (1, -3):
            old = _old_per_group_shift(vals, gids, shift)
            new = per_group_shift(vals, gids, shift)
            # identity (NaN-aware)
            ident = np.array_equal(old, new, equal_nan=True)
            t_old = _bestof(_old_per_group_shift, (vals, gids, shift))
            t_new = _bestof(per_group_shift, (vals, gids, shift))
            print(
                f"n={n:>9} groups={n_groups:>7} shift={shift:>2} "
                f"OLD={t_old*1e3:8.2f}ms NEW={t_new*1e3:8.2f}ms "
                f"speedup={t_old/t_new:5.2f}x identical={ident}"
            )


if __name__ == "__main__":
    main()
