"""Benchmark: is the flat homogeneity_tau=0.6 PC1-variance gate miscalibrated at
small k? (audit cluster-aggregate-4). The gate accepts a cluster as
"unidimensional" (safe to collapse to one aggregate) iff
    var_ratio = sv[0]^2 / sum(sv^2) >= homogeneity_tau (0.6).
Concern: a genuine 2-FACTOR cluster (members loading on two distinct latents)
at small k may still clear 0.6 -> wrongly collapsed, losing the 2nd factor.

We measure var_ratio for 1-factor vs 2-factor clusters across k, and compare the
flat 0.6 cut to a Horn parallel-analysis cut (95th pct of var_ratio for k
INDEPENDENT standardized Gaussians at the same n,k). A WIN for k-aware Horn iff
it rejects 2-factor clusters the flat 0.6 wrongly accepts, while still accepting
true 1-factor clusters.

RESULT (2026-06-03): NO ACTIONABLE WIN, and the proposed Horn fix is WRONG.
1-factor var_ratio 0.88-0.91 (accepted); 2-factor var_ratio 0.45-0.55 for k>=4
(flat 0.6 correctly REJECTS). The only mis-accept is k=3 (2-factor=0.621, barely
over 0.6). Horn parallel-analysis is the WRONG criterion: its cut (0.35 at k=3
down to 0.10 at k=12) ACCEPTS even the 2-factor clusters, because Horn tests
"is there ANY shared structure" not "is it UNIDIMENSIONAL" -- adopting it would
make the gate accept MORE multi-factor blobs. Keep the flat 0.6; the k=3 edge is
marginal and in an off-default-path discovery step.
"""
from __future__ import annotations

import numpy as np


def _var_ratio(M):
    Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1.0)
    Zc = Z - Z.mean(0)
    sv = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
    return float(sv[0] ** 2 / max(np.sum(sv ** 2), 1e-12))


def _horn_cut(n, k, seed, reps=200, pct=95):
    rng = np.random.default_rng(seed)
    vals = [_var_ratio(rng.standard_normal((n, k))) for _ in range(reps)]
    return float(np.percentile(vals, pct))


def main():
    n = 2000
    tau = 0.6
    print(f"n={n}  flat homogeneity_tau={tau}")
    print(f"{'k':>3} | {'1factor vr':>10} {'2factor vr':>10} | {'Horn cut':>9} | "
          f"{'flat verdict (2f)':>17} {'Horn verdict (2f)':>17}")
    for k in (3, 4, 5, 6, 8, 12):
        vr1, vr2 = [], []
        for seed in range(12):
            rng = np.random.default_rng(seed * 17 + k)
            # 1-factor: all members reflect one latent.
            z = rng.standard_normal(n)
            M1 = np.column_stack([z + 0.4 * rng.standard_normal(n) for _ in range(k)])
            # 2-factor: two latents, members split (loadings on z1 OR z2), so PC1
            # cannot explain >~ half the standardized variance.
            z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
            cols = []
            for i in range(k):
                base = z1 if i % 2 == 0 else z2
                cols.append(base + 0.4 * rng.standard_normal(n))
            M2 = np.column_stack(cols)
            vr1.append(_var_ratio(M1)); vr2.append(_var_ratio(M2))
        vr1m, vr2m = np.mean(vr1), np.mean(vr2)
        horn = _horn_cut(n, k, seed=k)
        flat_2f = "ACCEPT(bad)" if vr2m >= tau else "reject(ok)"
        horn_2f = "ACCEPT(bad)" if vr2m >= horn else "reject(ok)"
        print(f"{k:>3} | {vr1m:>10.3f} {vr2m:>10.3f} | {horn:>9.3f} | "
              f"{flat_2f:>17} {horn_2f:>17}")
    print("\nWIN for Horn iff it says reject(ok) on 2-factor where flat says "
          "ACCEPT(bad); NO win if both agree.")


if __name__ == "__main__":
    main()
