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

import numpy as np

from mlframe._bench_timing_shared import best_of_seconds_args_no_warmup
from mlframe.feature_engineering.grouped import per_group_shift

# prod's own per_group_shift IS the OLD segment-loop (the vectorized rewrite below was
# bench-rejected and never shipped -- see grouped.py::per_group_shift's docstring comment).
_old_per_group_shift = per_group_shift


_bestof = best_of_seconds_args_no_warmup


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
