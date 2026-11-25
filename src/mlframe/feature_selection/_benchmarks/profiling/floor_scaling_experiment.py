"""Is the mm-debiased order-2 maxT floor of n rows well estimated by n_sub / n times the floor of a subsample of n_sub rows?

Under the null the debiased plug-in MI of a pair is (chi2 - df) / (2 n) to first order, so its maximum over the pair pool scales as 1 / n. Compare, over pools of random columns (no signal), the exact floor at
n rows (what the fit computes: mm_debias=True takes every row) with the scaled floor of subsamples. Prints the ratio scaled / exact per configuration and seed.
"""
import sys
import time

import numpy as np

from mlframe.feature_selection.filters import _permutation_null as pn

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
n_cols = int(sys.argv[2]) if len(sys.argv) > 2 else 120
subs = [int(v) for v in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["15000", "30000"])]
K = 25
print(f"n={n} columns={n_cols} (pairs {n_cols * (n_cols - 1) // 2}) K={K} subsample sizes {subs}")
print(f"{'bins':>4s} {'ky':>3s} {'seed':>4s} {'exact floor':>12s} " + " ".join(f"{'scaled@' + str(s):>14s} {'ratio':>6s}" for s in subs) + "   secs(exact)")
rows = []
for bins in (8, 16, 32):
    for ky in (2, 4):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed + 10 * bins + ky)
            data = rng.integers(0, bins, (n, n_cols)).astype(np.int8)
            nb = np.full(n_cols, bins, dtype=np.int64)
            y = rng.integers(0, ky, n).astype(np.int64)
            ia, ib = np.triu_indices(n_cols, 1)
            ia, ib = ia.astype(np.int64), ib.astype(np.int64)

            def floor_of(rows_n):
                """The mm-debiased floor over every pair on the first ``rows_n`` rows."""
                d = np.ascontiguousarray(data[:rows_n])
                yy = np.ascontiguousarray(y[:rows_n])
                fy = np.bincount(yy, minlength=ky) / rows_n
                pn._PAIR_FLOOR_MEMO.clear() if hasattr(pn._PAIR_FLOOR_MEMO, "clear") else None
                return pn.pooled_pair_permutation_null_joint_mi_floor(
                    factors_data=d, nbins=nb, pair_a=ia, pair_b=ib, classes_y=yy, freqs_y=fy, n_permutations=K, quantile=0.95, random_seed=seed, mm_debias=True
                )

            t0 = time.perf_counter()
            exact = floor_of(n)
            t_exact = time.perf_counter() - t0
            out = []
            for s in subs:
                scaled = floor_of(s) * s / n
                out.append((scaled, scaled / exact if exact > 0 else float("nan")))
            rows.append((bins, ky, seed, exact, out))
            print(f"{bins:4d} {ky:3d} {seed:4d} {exact:12.3e} " + " ".join(f"{sc:14.3e} {r:6.2f}" for sc, r in out) + f"   {t_exact:5.1f}")
for j, s in enumerate(subs):
    ratios = np.array([r[4][j][1] for r in rows])
    print(f"subsample {s}: ratio scaled/exact mean {np.nanmean(ratios):.3f}, sd {np.nanstd(ratios):.3f}, min {np.nanmin(ratios):.2f}, max {np.nanmax(ratios):.2f}")
