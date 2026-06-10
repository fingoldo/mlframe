"""Calibrate a HARDER redundancy fixture where SelectKBest provably wastes budget
on copies of the dominant cluster. One cluster has stronger marginal signal (more
copies / tighter corr / bigger weight in y) so MI top-4 fills with its duplicates,
while MRMR's redundancy gate diversifies across clusters + independents."""
import warnings, numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def make_dominant_cluster_fixture(n=2000, seed=0):
    """One DOMINANT cluster (6 tight copies, strongest y-weight) + 2 weaker
    clusters (3 copies each) + 2 independent signals + 20 noise. SelectKBest's
    top-4 by marginal MI is fooled into picking ~4 copies of the dominant cluster
    (all near-identical -> redundant), leaving the weaker clusters + independents
    unrepresented, which caps downstream AUC."""
    rng = np.random.default_rng(seed)
    zc = [rng.standard_normal(n) for _ in range(3)]
    cols = {}
    copies = [6, 3, 3]
    tight = [0.05, 0.25, 0.25]
    for c in range(3):
        for k in range(copies[c]):
            cols[f"clu{c}_m{k}"] = zc[c] + tight[c] * rng.standard_normal(n)
    i0 = rng.standard_normal(n); i1 = rng.standard_normal(n)
    cols["indep0"] = i0; cols["indep1"] = i1
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    # dominant cluster has the LARGEST single-feature marginal signal but the
    # OTHER clusters + independents carry complementary info; an oracle needs
    # one-per-cluster + the independents.
    score = 2.2 * zc[0] + 1.3 * zc[1] + 1.3 * zc[2] + 1.1 * i0 + 1.1 * i1 + 0.3 * rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def auc_cols(X, y, cols, cv=5):
    if len(cols) == 0:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def fit_mrmr(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(cv=3, run_additional_rfecv_minutes=False, random_seed=seed,
                   min_features_fallback=1, verbose=0, use_simple_mode=False,
                   dcd_enable=True, dcd_tau_cluster=0.4, dcd_cluster_size_threshold=3,
                   min_relevance_gain=0.001, full_npermutations=10).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


print("=== dominant-cluster fixture K=4 ===")
for seed in (0, 1, 2):
    X, y = make_dominant_cluster_fixture(n=2000, seed=seed)
    K = 4
    skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
    skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
    order = fit_mrmr(X, y, seed)
    m4 = order[:K]
    # count how many of skb's picks are from the dominant cluster0 (the trap)
    skb_clu0 = sum(1 for c in skb_cols if c.startswith("clu0_"))
    m_clusters = len({c.split("_")[0] for c in m4 if c.startswith("clu")})
    a_m4 = auc_cols(X, y, m4); a_skb = auc_cols(X, y, skb_cols)
    a_rfe = auc_cols(X, y, [X.columns[i] for i in np.flatnonzero(
        RFE(LogisticRegression(max_iter=400), n_features_to_select=K).fit(X.values, y.values).get_support())])
    rng = np.random.default_rng(7000 + seed)
    a_rand = float(np.mean([auc_cols(X, y, list(rng.choice(X.columns, K, replace=False))) for _ in range(5)]))
    a_all = auc_cols(X, y, list(X.columns))
    print(f"seed={seed} |full|={len(order)} m4={m4} mclu={m_clusters}")
    print(f"   skb_cols={skb_cols} skb_clu0={skb_clu0}")
    print(f"   AUC m4={a_m4:.4f} skb={a_skb:.4f} rfe={a_rfe:.4f} rand={a_rand:.4f} all={a_all:.4f} "
          f"d(m4-skb)={a_m4-a_skb:+.4f}")
