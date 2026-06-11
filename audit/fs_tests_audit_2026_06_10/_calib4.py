"""Find an MRMR config that produces a diversified set on the dominant-cluster
fixture (raise patience so it explores past the dominant cluster's redundant copies)."""
import warnings, numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def make_fix(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    zc = [rng.standard_normal(n) for _ in range(3)]
    cols = {}
    copies = [6, 3, 3]; tight = [0.05, 0.25, 0.25]
    for c in range(3):
        for k in range(copies[c]):
            cols[f"clu{c}_m{k}"] = zc[c] + tight[c] * rng.standard_normal(n)
    i0 = rng.standard_normal(n); i1 = rng.standard_normal(n)
    cols["indep0"] = i0; cols["indep1"] = i1
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    score = 2.2 * zc[0] + 1.3 * zc[1] + 1.3 * zc[2] + 1.1 * i0 + 1.1 * i1 + 0.3 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def auc_cols(X, y, cols, cv=5):
    if len(cols) == 0:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def fit_mrmr(X, y, seed, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(cv=3, run_additional_rfecv_minutes=False, random_seed=seed,
                   verbose=0, use_simple_mode=False, **kw).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


configs = {
    "patient_dcd": dict(dcd_enable=True, dcd_tau_cluster=0.4, dcd_cluster_size_threshold=3,
                        min_relevance_gain=0.0, min_relevance_gain_relative_to_first=0.0,
                        max_consec_unconfirmed=40, full_npermutations=10, min_features_fallback=4),
    "patient_nodcd": dict(min_relevance_gain=0.0, min_relevance_gain_relative_to_first=0.0,
                          max_consec_unconfirmed=40, full_npermutations=10, min_features_fallback=4),
}
for cfgname, kw in configs.items():
    print(f"\n=== {cfgname} ===")
    for seed in (0, 1, 2):
        X, y = make_fix(n=2000, seed=seed)
        K = 4
        skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
        skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
        order = fit_mrmr(X, y, seed, **kw)
        m4 = order[:K]
        mclu = len({c.split("_")[0] for c in m4 if c.startswith("clu")})
        a_m4 = auc_cols(X, y, m4); a_skb = auc_cols(X, y, skb_cols)
        a_mfull = auc_cols(X, y, order)
        print(f"  seed={seed} |full|={len(order)} m4={m4} mclu={mclu} "
              f"AUC m4={a_m4:.4f} mfull={a_mfull:.4f} skb={a_skb:.4f} d4={a_m4-a_skb:+.4f}")
