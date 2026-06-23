"""Bench: RankGauss FIT-path rank computation -- two-sweep searchsorted vs argsort self-rank.

At fit time ``generate_rankgauss_features`` ranks the fit values among themselves. The original code did this with
``_avg_tie_rank(np.sort(x), x)`` -- two full-array ``np.searchsorted`` sweeps (side left + right) -- which is just
computing the array's own ranks the expensive way. ``_self_avg_tie_rank`` computes the identical average (mid) ranks
with a single ``argsort`` + a run-length mid-rank scatter, no searchsorted. Bit-identical on continuous AND tied data;
the replay (apply) path is untouched (there the test values are NOT the fit values, so searchsorted is still required).

Run:  CUDA_VISIBLE_DEVICES="" python bench_rankgauss_fit_self_rank.py

Measured (this dev box, python 3.14.3, best-of):
  isolated self-rank @ n=1M:  searchsorted 1.01s  ->  argsort 0.30s   (~3.4x), bit-identical
  full generate_rankgauss_features, 6 cols x 1M rows:  6.72s -> 2.58s  (~2.6x)
Selection impact: NONE -- RankGauss is monotone (DPI), not MI-gated; output is bit-identical so any downstream
linear/NN lift is unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._extra_fe_families import (
    _avg_tie_rank,
    _self_avg_tie_rank,
    generate_rankgauss_features,
)


def _old_self_rank(x: np.ndarray) -> np.ndarray:
    return _avg_tie_rank(np.sort(x), x)


def _best(fn, *a, reps: int = 5) -> float:
    fn(*a)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> None:
    rng = np.random.default_rng(0)

    print("== isolated self-rank (identity + timing) ==")
    for desc, x in [
        ("cont_1M", rng.standard_normal(1_000_000)),
        ("disc_1M", rng.integers(0, 50, 1_000_000).astype(np.float64)),
    ]:
        old = _old_self_rank(x)
        new = _self_avg_tie_rank(x)
        ok = np.array_equal(old, new)
        t_old = _best(_old_self_rank, x)
        t_new = _best(_self_avg_tie_rank, x)
        print(
            f"  {desc:10s} identical={ok}  old={t_old*1e3:7.1f}ms  new={t_new*1e3:7.1f}ms  "
            f"speedup={t_old/t_new:4.2f}x"
        )

    print("== full generate_rankgauss_features (6 cols x 1M) ==")
    num = pd.DataFrame({f"x{i}": rng.normal(size=1_000_000) for i in range(6)})
    t = _best(generate_rankgauss_features, num, list(num.columns), reps=3)
    print(f"  generate (NEW) best={t:.3f}s")


if __name__ == "__main__":
    main()
