"""Benchmark: does the Pearson-gated cluster detector MISS monotone-nonlinear
reflection clusters that a rank (Spearman) gate would catch? (audit
cluster-aggregate-5). The post-hoc _discover_clusters and the orth-basis
detect_clusters_by_correlation link on Pearson |corr| >= threshold. A cluster of
monotone-nonlinear reflections of one latent (z, z^3, exp, steep-sigmoid, rank)
shares all its information but can have Pearson well below threshold while
Spearman/MI stays high -> the genuine redundancy is missed.

We plant ONE monotone-nonlinear reflection group + a second independent group +
noise, cluster via single-linkage on Pearson vs Spearman at the same threshold,
and report recovery (ARI vs planted truth). WIN for Spearman iff it recovers the
nonlinear group Pearson splits, without spuriously merging the two groups.

RESULT (2026-06-03): NO actionable gap. Pearson recovers the monotone-nonlinear
z-group INTACT (ARI 1.0 = Spearman). Two reasons the Pearson gate is fine for
its purpose: (1) single-linkage CHAINS the group through its near-linear members
(rank(z)/sigmoid keep Pearson >0.6 with the rest), so one hub holds it together;
(2) the only case Pearson truly misses is NON-monotone (z vs z^2: Pearson ~0,
MI high) -- but z^2 is not a noisy copy of z, so it should NOT join z's DENOISING
cluster (that is the orth-basis FE's job, not cluster aggregation). Switching the
edge to Spearman/MI changes nothing here. Keep Pearson.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _single_linkage(C, tau):
    p = C.shape[0]; parent = list(range(p))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(p):
        for j in range(i + 1, p):
            if abs(C[i, j]) >= tau:
                parent[find(j)] = find(i)
    roots = [find(i) for i in range(p)]
    u = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([u[r] for r in roots])


def _make(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    eps = lambda s=0.05: s * rng.standard_normal(n)
    cols = {
        # monotone-NONLINEAR reflections of z (all share z's information):
        "z_lin": z + eps(),
        "z_cube": np.sign(z) * np.abs(z) ** 3 + eps(),
        "z_exp": np.exp(1.5 * z) + eps(),
        "z_sig": 1.0 / (1.0 + np.exp(-6.0 * z)) + eps(),
        "z_rank": pd.Series(z).rank().to_numpy() + eps(1.0),
        # second, independent latent group (linear) -- must NOT merge with z group:
        "w0": w + eps(0.2), "w1": w + eps(0.2), "w2": w + eps(0.2),
        # pure noise:
        "n0": rng.standard_normal(n), "n1": rng.standard_normal(n),
    }
    truth = np.array([0, 0, 0, 0, 0, 1, 1, 1, -1, -1])
    return pd.DataFrame(cols), truth


def main():
    from sklearn.metrics import adjusted_rand_score
    tau = 0.6
    print(f"tau={tau}; planted: 5 monotone-nonlinear reflections of z + 3 linear " f"reflections of w + 2 noise")
    pe, sp = [], []
    for seed in range(6):
        X, truth = _make(3000, seed)
        mask = truth >= 0
        Cp = np.nan_to_num(X.corr(method="pearson").to_numpy())
        Cs = np.nan_to_num(X.corr(method="spearman").to_numpy())
        lp = _single_linkage(Cp, tau)
        ls = _single_linkage(Cs, tau)
        pe.append(adjusted_rand_score(truth[mask], lp[mask]))
        sp.append(adjusted_rand_score(truth[mask], ls[mask]))
        # show how many of the 5 z-reflections Pearson keeps in one cluster
        z_idx = list(range(5))
        pe_groups = len(set(lp[z_idx].tolist()))
        sp_groups = len(set(ls[z_idx].tolist()))
        print(f"seed={seed}: Pearson ARI={pe[-1]:.3f} (z-group split into {pe_groups}) | "
              f"Spearman ARI={sp[-1]:.3f} (z-group split into {sp_groups})  [1=intact]")
    print("---")
    print(f"mean ARI: Pearson={np.mean(pe):.3f}  Spearman={np.mean(sp):.3f}  " f"delta={np.mean(sp)-np.mean(pe):+.3f}")
    print("WIN for Spearman/MI edge iff it keeps the nonlinear z-group intact " "where Pearson splits it.")


if __name__ == "__main__":
    main()
