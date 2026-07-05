"""Benchmark: on OUTLIER-contaminated reflection clusters, does a Ledoit-Wolf
shrunk / robust PC1 recover the clean latent better than the EXISTING menu
(which already includes the outlier-robust median_z)? (audit
dcd-pca-shrinkage-10 / dcd-core-8). Clean-Gaussian benchmarks can't show
shrinkage value; the audit specifically named the outlier regime.

Metric: |corr(aggregate, clean latent z)| -- how well the aggregate recovers the
true signal despite outliers. WIN for LW iff it beats the best existing combiner
(mean_z / median_z / pca_pc1) by a material margin. If median_z already matches
LW, the existing menu suffices and shrinkage adds nothing.

RESULT (2026-06-03): NO WIN, and Ledoit-Wolf is the WRONG tool. Under 2-10%
outliers mean_z/pca_pc1 degrade (|corr| 0.98 -> 0.80) but median_z stays robust
at 0.977 across ALL levels, while LW_pc1 == pca_pc1 (0.80) -- LW shrinks the
covariance for small-n instability, NOT for outliers, so its top eigenvector is
just as corrupted. The best existing combiner is median_z and LW is -0.04..-0.18
WORSE. The DCD auto bake-off already includes median_z and selects by OOF MI, so
it picks the robust combiner when outliers bite. No new combiner needed.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._cluster_aggregate import (
    _apply_method_nonlinear,
    _derive_weights,
    _standardize_align,
)


def _agg(Z, method):
    w = _derive_weights(Z, method)
    return _apply_method_nonlinear(Z, method) if w is None else (Z @ np.asarray(w))


def _lw_pc1(Z):
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf().fit(Z).covariance_
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    v = v * np.sign(v[int(np.argmax(np.abs(v)))] or 1.0)
    return Z @ v


def main():
    print("metric = |corr(aggregate, clean latent)|; higher = better recovery")
    print(f"{'outlier%':>8} | {'mean_z':>7} {'median_z':>8} {'pca_pc1':>7} | " f"{'LW_pc1':>7} | {'LW - best_existing':>18}")
    for frac in (0.0, 0.02, 0.05, 0.10):
        res = {m: [] for m in ("mean_z", "median_z", "pca_pc1", "lw")}
        for seed in range(10):
            rng = np.random.default_rng(seed * 7 + int(frac * 1000))
            n, k = 3000, 5
            z = rng.standard_normal(n)
            M = np.column_stack([z + 0.4 * rng.standard_normal(n) for _ in range(k)])
            # Inject outliers: a fraction of rows get extreme spikes in 1-2 members.
            n_out = int(frac * n)
            if n_out:
                rows = rng.choice(n, n_out, replace=False)
                for r in rows:
                    j = rng.integers(0, k)
                    M[r, j] += rng.choice([-1, 1]) * rng.uniform(8, 15)
            Z, *_ = _standardize_align(M, 0)
            for m in ("mean_z", "median_z", "pca_pc1"):
                res[m].append(abs(np.corrcoef(_agg(Z, m), z)[0, 1]))
            res["lw"].append(abs(np.corrcoef(_lw_pc1(Z), z)[0, 1]))
        mz, md, pc, lw = (np.mean(res[m]) for m in ("mean_z", "median_z", "pca_pc1", "lw"))
        best_exist = max(mz, md, pc)
        print(f"{frac*100:>7.0f}% | {mz:>7.3f} {md:>8.3f} {pc:>7.3f} | {lw:>7.3f} | " f"{lw - best_exist:>+18.3f}")
    print("WIN for LW iff the last column is materially > 0 (LW beats best existing).")


if __name__ == "__main__":
    main()
