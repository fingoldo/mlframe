"""CPX15 bench: GroupTimeSeriesSplit.split per-fold train accumulation.

The hot loop rebuilds ``np.array(sorted(set(train_buf)))`` over a GROWING ordered prefix of groups every fold.
This bench times the full ``list(split(...))`` on a realistic shape (n_samples=200k, many groups, n_splits=10),
best-of-N warm, and is used to A/B the incremental-accumulation rewrite against the prior code.

Run:  set CUDA_VISIBLE_DEVICES="" ;  python src/mlframe/models/_benchmarks/bench_cpx15_selection.py
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.models.selection import GroupTimeSeriesSplit


def make_groups(n_samples: int, n_groups: int, *, interleave: bool, seed: int = 0) -> np.ndarray:
    """Build a (n_samples,) integer group label array preserving first-appearance group order.

    interleave=False: contiguous time-ordered blocks (the typical time-series case, group indices contiguous).
    interleave=True:  groups interleaved so each group's index list is NOT a contiguous slice (stresses the global sort).
    """
    rng = np.random.default_rng(seed)
    if not interleave:
        # contiguous blocks of (roughly) equal size, in order 0,1,2,...
        bounds = np.linspace(0, n_samples, n_groups + 1).astype(np.int64)
        groups = np.empty(n_samples, dtype=np.int64)
        for g in range(n_groups):
            groups[bounds[g] : bounds[g + 1]] = g
        return groups
    # interleaved: assign each sample a group, but keep first-appearance order == 0..n_groups-1
    groups = rng.integers(0, n_groups, size=n_samples).astype(np.int64)
    # force group g to first-appear before g+1 by stamping a header block of 0..n_groups-1 at the front
    groups[:n_groups] = np.arange(n_groups)
    return groups


def time_split(n_samples: int, n_groups: int, n_splits: int, *, interleave: bool, best_of: int = 7) -> float:
    groups = make_groups(n_samples, n_groups, interleave=interleave)
    splitter = GroupTimeSeriesSplit(n_splits=n_splits)
    # warm
    _ = list(splitter.split(groups, groups=groups))
    best = float("inf")
    for _ in range(best_of):
        t0 = time.perf_counter()
        _ = list(splitter.split(groups, groups=groups))
        best = min(best, time.perf_counter() - t0)
    return best


def _ab():
    """A/B the live (incremental-merge) split() against the committed pre-CPX15 baseline (_old_selection_cpx15.py)."""
    from mlframe.models._benchmarks._old_selection_cpx15 import GroupTimeSeriesSplit as OLD
    import time as _t

    def best(cls, groups, n=9):
        sp = cls(n_splits=10)
        list(sp.split(groups, groups=groups))
        b = float("inf")
        for _ in range(n):
            t0 = _t.perf_counter()
            list(sp.split(groups, groups=groups))
            b = min(b, _t.perf_counter() - t0)
        return b * 1e3

    for interleave in (False, True):
        for n_groups in (500, 5000):
            g = make_groups(200_000, n_groups, interleave=interleave)
            o, nw = best(OLD, g), best(GroupTimeSeriesSplit, g)
            print(f"interleave={interleave!s:5} n_groups={n_groups}: OLD={o:7.2f}ms NEW={nw:7.2f}ms  speedup={o/nw:.2f}x")


if __name__ == "__main__":
    for interleave in (False, True):
        for n_samples, n_groups, n_splits in ((200_000, 500, 10), (200_000, 5000, 10)):
            t = time_split(n_samples, n_groups, n_splits, interleave=interleave)
            print(f"interleave={interleave!s:5} n={n_samples} n_groups={n_groups} n_splits={n_splits}: best={t*1e3:.2f} ms")
    print("--- A/B vs pre-CPX15 baseline ---")
    _ab()
