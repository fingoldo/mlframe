"""Benchmark: does the post-hoc cluster_aggregate richer menu beat mean_z-only?
(audit cluster-aggregate-8 / integration-defaults-12). The default menu is
('mean_z',). On a HETEROSCEDASTIC cluster (members with very different noise
levels) an inverse-variance / PCA combiner should denoise better than a flat
mean. NOTE the post-hoc path only runs when dcd_enable=False (with DCD on,
dcd_swap_method='auto' already runs the 7-method bake-off), so we test that
regime. Measures OOS downstream AUC: mean_z-only vs the curated menu.

RESULT (2026-06-03): NO ACTIONABLE WIN -> keep the ('mean_z',) default. On
heteroscedastic clusters the bake-off picks a different combiner (factor_score /
mean_inv_var) in 2/6 seeds, but mean OOS AUC is identical to mean_z-only
(+0.0000). The post-hoc path is also off by default (DCD on -> dcd_swap_method=
'auto' already runs the 7-method per-cluster bake-off during screening).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _hetero_frame(n, seed):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    strong = rng.standard_normal(n)
    # Heteroscedastic reflections: a couple clean, several very noisy. mean_z
    # over-weights the noisy ones; inv-var / PC1 should downweight them.
    sigmas = [0.2, 0.3, 1.3, 1.6, 1.9]
    refl = {f"refl{i}": latent + s * rng.standard_normal(n) for i, s in enumerate(sigmas)}
    X = pd.DataFrame({"strong": strong, **refl,
                      "noise_a": rng.standard_normal(n), "noise_b": rng.standard_normal(n)})
    logit = 1.2 * strong + 1.7 * latent
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, pd.Series(y)


def _fit_score(methods, Xtr, ytr, Xte, yte):
    from mlframe.feature_selection.filters.mrmr import MRMR
    m = MRMR(
        dcd_enable=False,  # post-hoc cluster_aggregate path
        cluster_aggregate_enable=True, cluster_aggregate_mode="replace",
        cluster_aggregate_methods=methods, cluster_aggregate_corr_threshold=0.5,
        verbose=0, random_seed=0,
    ).fit(Xtr, ytr)
    clf = LogisticRegression(max_iter=2000).fit(m.transform(Xtr), ytr)
    auc = roc_auc_score(yte, clf.predict_proba(m.transform(Xte))[:, 1])
    names = [c for c in m.get_feature_names_out()]
    agg = [c for c in names if "clusteragg" in c]
    return auc, (agg[0] if agg else "none")


def main():
    menu = ("mean_z", "mean_inv_var", "pca_pc1", "factor_score")
    a_mean, a_menu = [], []
    for seed in range(6):
        X, y = _hetero_frame(3000, seed)
        Xtr, ytr, Xte, yte = X.iloc[:2000], y.iloc[:2000], X.iloc[2000:], y.iloc[2000:]
        am, win_m = _fit_score(("mean_z",), Xtr, ytr, Xte, yte)
        amenu, win_menu = _fit_score(menu, Xtr, ytr, Xte, yte)
        a_mean.append(am); a_menu.append(amenu)
        print(f"seed={seed}: mean_z auc={am:.4f} ({win_m}) | menu auc={amenu:.4f} ({win_menu})")
    print("---")
    print(f"mean OOS AUC: mean_z-only={np.mean(a_mean):.4f}  curated-menu={np.mean(a_menu):.4f}  "
          f"delta={np.mean(a_menu)-np.mean(a_mean):+.4f}")


if __name__ == "__main__":
    main()
