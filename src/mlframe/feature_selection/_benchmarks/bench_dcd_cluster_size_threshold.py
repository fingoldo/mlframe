"""Benchmark: DCD cluster_size_threshold 4 vs 2 (audit dcd-core-3 / dcd-swap-threshold-3).

Now that the swap permutation-null is fixed (dcd_swap_npermutations decoupled
from full_npermutations), the canonical 3-feature redundancy cluster (anchor +
2 noisy reflections of a latent that drives y) can actually swap to a denoised
aggregate at threshold=2 -- but threshold=4 never reaches the swap gate, so it
keeps the raw (noisy) anchor.

Hypothesis: at threshold=2 the denoised aggregate is a cleaner feature -> equal-
or-better out-of-sample downstream quality, WITHOUT shrinking support_.

Bounded n (<=4000) / small p to respect memory. Prints OOS AUC, support size,
and n_swaps for both thresholds across seeds.

RESULT (2026-06-03, after the swap-null fix): NO ACTIONABLE WIN -> keep default 4.
Even with reflections tuned to cluster (tau=0.4, noise=0.3, pruned 3-4 members),
threshold=2 fired a swap in only 1/5 seeds and mean OOS AUC moved +0.0009
(noise-level); support identical. The binding constraint is the swap GATE
(aggregate must beat the anchor's conditional MI by swap_gain_threshold AND pass
the permutation null), not the size threshold -- and that gate is correctly
conservative on small clusters. Lowering the default is unjustified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _make_frame(n, seed):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    strong = rng.standard_normal(n)
    # 4 reflections of latent (anchor + 3 members => clusters at threshold>=2 but
    # NOT threshold=4). Low per-member noise so binned SU clears tau (clustering),
    # while the mean-of-4 still cuts the residual noise variance ~4x vs a single
    # member -> a cleaner proxy for the y-driving latent.
    noise = 0.3
    refl = {f"refl{i}": latent + noise * rng.standard_normal(n) for i in range(4)}
    X = pd.DataFrame({
        "strong": strong,
        **refl,
        "noise_a": rng.standard_normal(n),
        "noise_b": rng.standard_normal(n),
    })
    logit = 1.3 * strong + 1.6 * latent
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return X, pd.Series(y)


def _fit_and_score(threshold, Xtr, ytr, Xte, yte):
    from mlframe.feature_selection.filters.mrmr import MRMR
    m = MRMR(
        dcd_enable=True, dcd_tau_cluster=0.4,
        dcd_cluster_size_threshold=threshold,
        verbose=0, random_seed=0,
    ).fit(Xtr, ytr)
    Xtr_sel = m.transform(Xtr)
    Xte_sel = m.transform(Xte)
    clf = LogisticRegression(max_iter=2000).fit(Xtr_sel, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte_sel)[:, 1])
    d = m.dcd_ or {}
    n_swaps = int(d.get("n_swaps", 0))
    n_pruned = int(d.get("n_pruned", 0))
    n_anchors = int(d.get("n_anchors", 0))
    n_feat = Xte_sel.shape[1]
    return auc, n_feat, n_swaps, n_pruned, n_anchors


def main():
    seeds = [0, 1, 2, 3, 4]
    rows = []
    for seed in seeds:
        X, y = _make_frame(3000, seed)
        ntr = 2000
        Xtr, ytr = X.iloc[:ntr], y.iloc[:ntr]
        Xte, yte = X.iloc[ntr:], y.iloc[ntr:]
        a4, f4, s4, p4, an4 = _fit_and_score(4, Xtr, ytr, Xte, yte)
        a2, f2, s2, p2, an2 = _fit_and_score(2, Xtr, ytr, Xte, yte)
        rows.append((seed, a4, f4, s4, a2, f2, s2))
        print(f"seed={seed}: thr4 auc={a4:.4f} nfeat={f4} swaps={s4} pruned={p4} anch={an4} | "
              f"thr2 auc={a2:.4f} nfeat={f2} swaps={s2} pruned={p2} anch={an2}")
    arr = np.array([(r[1], r[4], r[3], r[6], r[2], r[5]) for r in rows])
    print("---")
    print(f"mean OOS AUC: thr4={arr[:,0].mean():.4f}  thr2={arr[:,1].mean():.4f}  " f"delta(thr2-thr4)={arr[:,1].mean()-arr[:,0].mean():+.4f}")
    print(f"mean n_swaps: thr4={arr[:,2].mean():.1f}  thr2={arr[:,3].mean():.1f}")
    print(f"mean support: thr4={arr[:,4].mean():.1f}  thr2={arr[:,5].mean():.1f}")


if __name__ == "__main__":
    main()
