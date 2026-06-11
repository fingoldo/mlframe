import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def make_redundant_clusters(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(3)]
    cols = {}
    for c, lat in enumerate(latents):
        for k in range(4):
            cols[f"clu{c}_m{k}"] = lat + 0.1 * rng.standard_normal(n)
    indep1 = rng.standard_normal(n); indep2 = rng.standard_normal(n)
    cols["indep0"] = indep1; cols["indep1"] = indep2
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    score = sum(latents) + indep1 + indep2 + 0.3 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def auc_on_cols(X, y, cols, cv=5):
    if len(cols) == 0:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def fit_mrmr(X, y, seed, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(cv=3, run_additional_rfecv_minutes=False, random_seed=seed,
                   min_features_fallback=1, verbose=0, **kw).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)], sel


configs = {
    "dcd_tuned": dict(dcd_enable=True, dcd_tau_cluster=0.4, dcd_cluster_size_threshold=3,
                      min_relevance_gain=0.001, full_npermutations=10),
    "simple_mode": dict(use_simple_mode=True, min_relevance_gain=0.0, full_npermutations=10),
    "simple_dcd": dict(use_simple_mode=True, dcd_enable=True, dcd_tau_cluster=0.4,
                       dcd_cluster_size_threshold=3, min_relevance_gain=0.001, full_npermutations=10),
}

for cfgname, kw in configs.items():
    print(f"\n=== config={cfgname} ===")
    for seed in (0, 1, 2):
        X, y = make_redundant_clusters(n=2000, seed=seed)
        K = 4
        skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
        skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
        auc_skb = auc_on_cols(X, y, skb_cols)
        order, sel = fit_mrmr(X, y, seed, **kw)
        auc_m4 = auc_on_cols(X, y, order[:K])
        auc_mfull = auc_on_cols(X, y, order)
        # cluster coverage of top-4
        clusters = set()
        for nm in order[:K]:
            for c in range(3):
                if nm.startswith(f"clu{c}_"):
                    clusters.add(c)
        print(f"  seed={seed} |full|={len(order)} top4={order[:K]} clcov={len(clusters)} "
              f"AUC m4={auc_m4:.4f} mfull={auc_mfull:.4f} skb={auc_skb:.4f} d={auc_m4-auc_skb:+.4f}")
