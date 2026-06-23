"""A/B bench: per_group_cum_reduce(op="count") Python-loop (OLD) vs vectorized (NEW).

OLD: per group, allocate ``np.arange(1, seg_len+1)`` and scatter (Python loop over
every group). At many small groups (the prof_per_group_shift_10m shape: 200k groups,
~50 rows) the per-group Python iteration + 200k small arange allocations dominate.

NEW: compute the within-group 0-based rank of every sorted row in ONE vectorized pass
(``arange(n) - np.repeat(starts, seg_lens)``); count = rank+1 (or seg_len-rank when
reverse). No Python per-group loop, no per-group allocation. Bit-identical.

This wins only when groups are SMALL (avg <= ~30 rows). For few large groups the
Python loop is already trivial and the full-length repeat/rank temporaries make the
vectorized path ~2x SLOWER -- so production GATES on average group size
(_COUNT_VECTORIZE_MAX_AVG). This bench documents the crossover the gate is set from.

Run: CUDA_VISIBLE_DEVICES="" python bench_per_group_cum_count_vectorized_iter135.py
"""
import sys
sys.modules["cupy"] = None
import time

import numpy as np

from mlframe.feature_engineering.grouped import iter_group_segments, per_group_cum_reduce


def _old_count(group_ids, reverse=False, output_dtype=np.float64):
    n = len(group_ids)
    out = np.empty(n, dtype=output_dtype)
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        seg_idx = sort_idx[s:e]
        ar = np.arange(1, seg_idx.size + 1, dtype=output_dtype)
        if reverse:
            ar = ar[::-1]
        out[seg_idx] = ar
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
    print("=== representative shapes ===")
    for n, ng in [(1_000_000, 20_000), (10_000_000, 200_000), (1_000_000, 5)]:
        gids = rng.integers(0, ng, size=n).astype(np.int64)
        for rev in (False, True):
            old = _old_count(gids, rev)
            new = per_group_cum_reduce(np.empty(n), gids, "count", reverse=rev)
            ident = np.array_equal(old, new)
            t_old = _bestof(_old_count, (gids, rev))
            t_new = _bestof(
                lambda g, r: per_group_cum_reduce(np.empty(n), g, "count", reverse=r),
                (gids, rev),
            )
            print(
                f"n={n:>9} groups={ng:>7} avg={n // ng:>7} reverse={int(rev)} "
                f"OLD={t_old*1e3:7.2f}ms NEW={t_new*1e3:7.2f}ms "
                f"speedup={t_old/t_new:5.2f}x identical={ident}"
            )

    print("=== crossover sweep (n=2M, reverse=False) ===")
    n = 2_000_000
    for ng in [50, 200, 1000, 5000, 20000, 100000, 200000]:
        gids = rng.integers(0, ng, size=n).astype(np.int64)
        t_old = _bestof(_old_count, (gids, False))
        t_new = _bestof(
            lambda g, r: per_group_cum_reduce(np.empty(n), g, "count", reverse=r),
            (gids, False),
        )
        print(f"avg_group={n // ng:>7} OLD={t_old*1e3:7.2f} NEW={t_new*1e3:7.2f} speedup={t_old/t_new:5.2f}x")


if __name__ == "__main__":
    main()
