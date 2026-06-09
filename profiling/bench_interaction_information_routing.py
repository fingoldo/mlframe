"""cProfile + timing harness for the signed interaction-information routing (backlog idea #8).

Profiles the two hot paths:
  * ``pooled_pair_ii_null_floor`` -- the permutation null on max positive II over the candidate-pair pool
    (K shuffles x n_pairs x per-pair 3-MI re-score). This is the only added per-FE-step cost.
  * ``route_prospective_pairs`` -- the per-pair signed-II re-read + demotion (near-free: reads cached MIs).

Reports wall time + the cProfile top hotspots so the added cost is bounded and attributed. Run on a
RAM-contended box at n<=30000 (the user's constraint).
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/.claude/worktrees/mrmr-opt-campaign/src")

from mlframe.feature_selection.filters._interaction_information import (  # noqa: E402
    pooled_pair_ii_null_floor,
    route_prospective_pairs,
)

NBINS = 10


def _discretize(x, nbins=NBINS):
    ranks = np.argsort(np.argsort(x))
    return (ranks * nbins // len(x)).astype(np.int64)


def _build(n, p, seed=0):
    rng = np.random.default_rng(seed)
    factors = np.zeros((n, p), dtype=np.int64)
    for k in range(p):
        factors[:, k] = _discretize(rng.standard_normal(n))
    yc = _discretize(rng.standard_normal(n))
    nbins = np.array([NBINS] * p, dtype=np.int64)
    freqs_y = np.bincount(yc).astype(np.float64) / n
    import itertools
    pairs = list(itertools.combinations(range(p), 2))
    pa = np.array([q[0] for q in pairs], dtype=np.int64)
    pb = np.array([q[1] for q in pairs], dtype=np.int64)
    return factors, nbins, pa, pb, yc, freqs_y, pairs


def bench(n, p, k_perm=25):
    factors, nbins, pa, pb, yc, freqs_y, pairs = _build(n, p)
    ky = len(freqs_y)
    cached = {(i,): 0.05 for i in range(p)}
    pp = {((int(pa[i]), int(pb[i])), 0.1): 1 for i in range(len(pairs))}

    # warm + time the null floor
    t0 = time.perf_counter()
    floor = pooled_pair_ii_null_floor(
        factors_data=factors, nbins=nbins, pair_a=pa, pair_b=pb,
        marginal_mi_a=np.zeros(len(pairs)), marginal_mi_b=np.zeros(len(pairs)),
        classes_y=yc, freqs_y=freqs_y, n_permutations=k_perm, quantile=0.95, random_seed=0,
    )
    t_floor = time.perf_counter() - t0

    t0 = time.perf_counter()
    kept, routes, iis = route_prospective_pairs(
        pp, cached_MIs=cached, nbins=nbins, nbins_y=ky, n=n, ii_floor=floor, synergy_added_idx=set(),
    )
    t_route = time.perf_counter() - t0

    print(f"n={n} p={p} pairs={len(pairs)} K={k_perm}: null_floor={t_floor*1000:.1f}ms "
          f"route={t_route*1000:.2f}ms floor={floor:.5f}")
    return factors, nbins, pa, pb, yc, freqs_y, k_perm


def main():
    print("=== timing ===")
    for (n, p) in [(3000, 12), (10000, 20), (30000, 12), (30000, 25)]:
        bench(n, p)

    print("\n=== cProfile (n=30000, p=25, K=25) ===")
    factors, nbins, pa, pb, yc, freqs_y, k_perm = bench(30000, 25)
    pr = cProfile.Profile()
    pr.enable()
    pooled_pair_ii_null_floor(
        factors_data=factors, nbins=nbins, pair_a=pa, pair_b=pb,
        marginal_mi_a=np.zeros(pa.shape[0]), marginal_mi_b=np.zeros(pa.shape[0]),
        classes_y=yc, freqs_y=freqs_y, n_permutations=k_perm, quantile=0.95, random_seed=0,
    )
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
