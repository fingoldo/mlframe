"""Downstream A/B: ShapProxiedFS selection + OOS AUC with injected high-cardinality ID-like columns, chance correction off vs on."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def inject(X, seed, n_id=3, id_frac=1 / 3):
    rng = np.random.default_rng(seed + 100)
    n = len(X)
    X = X.copy()
    for k in range(n_id):
        X[f"idlike{k}"] = rng.integers(0, int(n * id_frac), n).astype(float)
    return X


def bins_for(X):
    bins, nb = {}, {}
    for c in X.columns:
        v = X[c].to_numpy()
        if c.startswith("idlike"):
            b = np.unique(v, return_inverse=True)[1].astype(np.int64)
        else:
            b = np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
        bins[c] = b
        nb[c] = int(b.max()) + 1
    return {"feature_names": list(X.columns), "su_to_target": np.full(X.shape[1], 0.1), "bins": bins, "nbins_per_feature": nb}


def run(X, y, chance, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed)
    pre = bins_for(Xtr)
    sps = ShapProxiedFS(
        random_state=seed, verbose=False, prefilter_top=30, max_features=6, n_models=1, n_splits=2, out_of_fold=False, revalidate=False,
        trust_guard=False, run_importance_ablation=False, cluster_features=True, cluster_auto_threshold=10, brute_force_max_features=12,
        shap_prefilter_enabled=False, precomputed=pre, cluster_su_chance_correction=chance,
    ).fit(Xtr, ytr)
    sel = list(sps.selected_features_)
    m = HistGradientBoostingClassifier(max_iter=80, random_state=0).fit(Xtr[sel], ytr)
    return sel, roc_auc_score(yte, m.predict_proba(Xte[sel])[:, 1]), sps.shap_proxy_report_["clustering"]


def main(seeds=range(5)):
    print("seed | same_sel | AUC old  AUC new | n_units old/new | id cols selected old/new")
    d = []
    for s in seeds:
        X, y, _ = make_regime_dataset(n_samples=3000, n_informative=5, n_redundant=5, redundancy_rho=0.8, n_noise=30, snr=8.0, task="binary", seed=s)
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X = inject(X, s)
        so, ao, co = run(X, y, False, s)
        sn, an, cn = run(X, y, True, s)
        d.append(an - ao)
        print(f"{s} | {set(so) == set(sn)} | {ao:.4f} {an:.4f} | {co.get('n_clusters', co.get('n_units'))}/{cn.get('n_clusters', cn.get('n_units'))} | "
              f"{sum(c.startswith('idlike') for c in so)}/{sum(c.startswith('idlike') for c in sn)}")
    print("mean dAUC", np.mean(d))


if __name__ == "__main__":
    main()
