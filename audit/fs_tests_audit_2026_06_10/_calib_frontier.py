"""Calibrate the quality-vs-K frontier (finding 03) and the honest signal_plus_noise
tie (finding 01b). MRMR selection-order prefix vs mutual_info_classif descending."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from feature_selection._biz_val_synth import make_signal_plus_noise, as_df


def make_dominant_cluster_fixture(n=2000, seed=0):
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


def mrmr_order(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(cv=3, run_additional_rfecv_minutes=False, random_seed=seed,
                   min_features_fallback=1, verbose=0, use_simple_mode=False,
                   dcd_enable=True, dcd_tau_cluster=0.4, dcd_cluster_size_threshold=3,
                   min_relevance_gain=0.001, full_npermutations=10).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


def mi_order(X, y, seed):
    mi = mutual_info_classif(X.values, y.values, random_state=seed)
    return [X.columns[i] for i in np.argsort(mi)[::-1]]


KS = [1, 2, 3, 4, 5, 6, 8]
print("=== frontier on dominant-cluster fixture ===")
for seed in (0, 1, 2):
    X, y = make_dominant_cluster_fixture(n=2000, seed=seed)
    m_ord = mrmr_order(X, y, seed)
    i_ord = mi_order(X, y, seed)
    row_m, row_i = {}, {}
    for K in KS:
        row_m[K] = auc_cols(X, y, m_ord[:K])
        row_i[K] = auc_cols(X, y, i_ord[:K])
    print(f"seed={seed} |mrmr|={len(m_ord)}")
    print("   K      :" + "".join(f"{K:>8}" for K in KS))
    print("   mrmr   :" + "".join(f"{row_m[K]:>8.4f}" for K in KS))
    print("   mi     :" + "".join(f"{row_i[K]:>8.4f}" for K in KS))
    diffs = [row_m[K] - row_i[K] for K in KS if not np.isnan(row_m[K])]
    print(f"   m[4]={row_m[4]:.4f} vs i[8]={row_i[8]:.4f} (eff d={row_m.get(4,float('nan'))-row_i[8]:+.4f})  "
          f"mean(m-i)={np.nanmean([row_m[K]-row_i[K] for K in KS]):+.4f}")

print("\n=== honest tie: signal_plus_noise (linear, no redundancy) K=4 ===")
for seed in (0, 1, 2):
    Xn, yn, sig = make_signal_plus_noise(n=1500, p_signal=4, p_noise=16, seed=seed)
    X, y = as_df(Xn, yn)
    K = 4
    skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
    skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
    m_ord = mrmr_order(X, y, seed)
    a_m = auc_cols(X, y, m_ord)  # MRMR full (may be < or > 4)
    a_m4 = auc_cols(X, y, m_ord[:K])
    a_skb = auc_cols(X, y, skb_cols)
    print(f"seed={seed} |mrmr|={len(m_ord)} mrmr_full_auc={a_m:.4f} mrmr_top4={a_m4:.4f} "
          f"skb={a_skb:.4f} d(mfull-skb)={a_m-a_skb:+.4f}")
