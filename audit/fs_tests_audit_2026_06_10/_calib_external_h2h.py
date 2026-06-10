"""Dev calibration for test_biz_val_h2h_external. Measures the actual AUC numbers
so the test floors can be pinned 5-15% below measured. NOT a committed test."""
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def make_redundant_clusters(n=2000, seed=0):
    """3 latent clusters x 4 noisy copies + 2 independent signals + 20 noise."""
    rng = np.random.default_rng(seed)
    latents = [rng.standard_normal(n) for _ in range(3)]
    cols = {}
    names_cluster = []
    for c, lat in enumerate(latents):
        for k in range(4):
            nm = f"clu{c}_m{k}"
            cols[nm] = lat + 0.1 * rng.standard_normal(n)
            names_cluster.append(nm)
    indep1 = rng.standard_normal(n)
    indep2 = rng.standard_normal(n)
    cols["indep0"] = indep1
    cols["indep1"] = indep2
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    # y depends on all 3 latents + 2 independent signals
    score = sum(latents) + indep1 + indep2 + 0.3 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y")


def auc_on_cols(X, y, cols, cv=5):
    if len(cols) == 0:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y,
                                 cv=cv, scoring="roc_auc").mean())


def mrmr_order(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False,
                   full_npermutations=10, random_seed=seed, min_features_fallback=1,
                   use_simple_mode=False, verbose=0).fit(X, y)
    names = list(X.columns)
    order = [names[i] for i in np.asarray(sel.support_, dtype=int)]
    return order, sel


print("=== Redundancy fixture: K=4 head-to-head (3 seeds) ===")
for seed in (0, 1, 2):
    X, y = make_redundant_clusters(n=2000, seed=seed)
    K = 4
    skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
    skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
    auc_skb = auc_on_cols(X, y, skb_cols)
    order, sel = mrmr_order(X, y, seed)
    mrmr_cols = order[:K]
    auc_mrmr_topk = auc_on_cols(X, y, mrmr_cols)
    auc_mrmr_full = auc_on_cols(X, y, order)
    rng = np.random.default_rng(1000 + seed)
    rand_aucs = []
    for _ in range(5):
        rc = list(rng.choice(X.columns, size=K, replace=False))
        rand_aucs.append(auc_on_cols(X, y, rc))
    auc_rand = float(np.mean(rand_aucs))
    auc_all = auc_on_cols(X, y, list(X.columns))
    rfe = RFE(LogisticRegression(max_iter=400), n_features_to_select=K).fit(X.values, y.values)
    rfe_cols = [X.columns[i] for i in np.flatnonzero(rfe.get_support())]
    auc_rfe = auc_on_cols(X, y, rfe_cols)
    print(f"seed={seed}: |mrmr_full|={len(order)} mrmr_top4_cols={mrmr_cols}")
    print(f"   AUC mrmr_top4={auc_mrmr_topk:.4f} skb={auc_skb:.4f} rfe={auc_rfe:.4f} "
          f"rand={auc_rand:.4f} all={auc_all:.4f} mrmr_full={auc_mrmr_full:.4f}")
    print(f"   skb_cols={skb_cols}")
    print(f"   delta mrmr-skb={auc_mrmr_topk-auc_skb:+.4f}")
