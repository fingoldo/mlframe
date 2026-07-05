"""Bench: row-subsample the order-2 Westfall-Young maxT pair-MI floor.

The order-2 maxT floor (``pooled_pair_permutation_null_joint_mi_floor``) is a
q=0.95 QUANTILE of the per-shuffle MAX joint-MI over the O(p^2) candidate-pair
pool, computed on the 30k screen subsample x K perms. It is a COARSE noise
threshold (the noise-floor-cap regime). This bench measures computing the floor
on a ROW-SUBSAMPLE of ``data`` (5k/10k/15k) vs full 30k:

  1. Speedup of the floor compute.
  2. SELECTION-equivalence: with observed pair-MI scored at FULL n (fixed),
     does the SAME set of pairs pass ``pair_mi >= floor`` under the subsampled
     floor as under the full-30k floor?

Run:  python -m mlframe.feature_selection._benchmarks.bench_pair_maxt_floor_subsample
"""

import time

import numpy as np

from mlframe.feature_selection.filters._permutation_null import (
    pooled_pair_permutation_null_joint_mi_floor,
)
from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    batch_pair_mi_prange,
)


def make_screen_data(n, p, nbins_val, n_signal_pairs, seed):
    """Synthetic ordinal screening matrix: p noise columns + a few synergistic
    (XOR-like) pairs whose JOINT MI clears any reasonable floor."""
    rng = np.random.default_rng(seed)
    ky = nbins_val
    data = rng.integers(0, nbins_val, size=(n, p)).astype(np.int64)
    classes_y = rng.integers(0, ky, size=n).astype(np.int64)
    # Plant synergy: y := (col0 + col1) % ky for the first pair, etc.
    for s in range(n_signal_pairs):
        a, b = 2 * s, 2 * s + 1
        if s == 0:
            classes_y = (data[:, a] + data[:, b]) % ky
        else:
            # partial signal on later pairs
            mask = rng.random(n) < 0.6
            classes_y[mask] = (data[mask, a] + data[mask, b]) % ky
    nbins = np.full(p, nbins_val, dtype=np.int64)
    freqs_y = np.bincount(classes_y, minlength=ky).astype(np.float64) / n
    return data, nbins, classes_y, freqs_y


def all_pairs(p):
    from itertools import combinations
    prs = list(combinations(range(p), 2))
    pa = np.fromiter((x[0] for x in prs), dtype=np.int64, count=len(prs))
    pb = np.fromiter((x[1] for x in prs), dtype=np.int64, count=len(prs))
    return pa, pb


def best_of(fn, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def main():
    N_FULL = 30000
    P = 60          # -> C(60,2)=1770 pairs
    NBINS = 10
    K = 25
    Q = 0.95
    SEED = 12345

    data, nbins, classes_y, freqs_y = make_screen_data(N_FULL, P, NBINS, n_signal_pairs=3, seed=SEED)
    pa, pb = all_pairs(P)
    n_pairs = len(pa)
    print(f"shape: n={N_FULL} p={P} n_pairs={n_pairs} K={K} q={Q} nbins={NBINS}")

    # Observed pair-MI at FULL n -- this is what the real gate compares to the floor.
    # Warm numba.
    _ = batch_pair_mi_prange(data[:64], pa, pb, nbins, classes_y[:64], freqs_y)
    obs_mi = batch_pair_mi_prange(data, pa, pb, nbins, classes_y, freqs_y)

    def floor_full():
        return pooled_pair_permutation_null_joint_mi_floor(
            factors_data=data, nbins=nbins, pair_a=pa, pair_b=pb,
            classes_y=classes_y, freqs_y=freqs_y,
            n_permutations=K, quantile=Q, random_seed=SEED,
        )

    # warm
    _ = pooled_pair_permutation_null_joint_mi_floor(
        factors_data=data[:2000], nbins=nbins, pair_a=pa, pair_b=pb,
        classes_y=classes_y[:2000], freqs_y=freqs_y, n_permutations=3, quantile=Q, random_seed=SEED,
    )

    t_full, floor_full_val = best_of(floor_full)
    survivors_full = set(np.nonzero(obs_mi >= floor_full_val)[0].tolist())
    print(f"\nFULL n={N_FULL}: floor={floor_full_val:.6f}  time={t_full*1000:.1f}ms  survivors={len(survivors_full)}")

    for cap in (5000, 10000, 15000, 20000):
        sub = data[:cap]
        sub_y = classes_y[:cap]
        sub_fy = np.bincount(sub_y, minlength=freqs_y.shape[0]).astype(np.float64) / cap

        def floor_sub():
            return pooled_pair_permutation_null_joint_mi_floor(
                factors_data=sub, nbins=nbins, pair_a=pa, pair_b=pb,
                classes_y=sub_y, freqs_y=sub_fy,
                n_permutations=K, quantile=Q, random_seed=SEED,
            )

        t_sub, floor_sub_val = best_of(floor_sub)
        survivors_sub = set(np.nonzero(obs_mi >= floor_sub_val)[0].tolist())
        added = survivors_sub - survivors_full
        dropped = survivors_full - survivors_sub
        eq = "IDENTICAL" if not added and not dropped else f"DIFF +{len(added)}/-{len(dropped)}"
        print(f"cap={cap:5d}: floor={floor_sub_val:.6f}  time={t_sub*1000:6.1f}ms "
              f"speedup={t_full/t_sub:4.2f}x  survivors={len(survivors_sub)}  selection={eq}")


if __name__ == "__main__":
    main()
