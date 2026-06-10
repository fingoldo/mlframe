"""Final calibration: patient_nodcd config across seeds 0-5 for (1) dominant-cluster
K=4 win, (2) full-frontier vs MI, (3) honest signal_plus_noise tie. Lock margins."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from feature_selection._biz_val_synth import make_signal_plus_noise, as_df

MRMR_KW = dict(min_relevance_gain=0.0, min_relevance_gain_relative_to_first=0.0,
               max_consec_unconfirmed=40, full_npermutations=10, min_features_fallback=4,
               use_simple_mode=False, dcd_enable=False, cv=3,
               run_additional_rfecv_minutes=False)


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


def mrmr_order(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(random_seed=seed, verbose=0, **MRMR_KW).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


KS = [1, 2, 3, 4, 5, 6, 8]
print("=== dominant-cluster K=4 win + frontier (seeds 0-5) ===")
win4 = eff = dom = 0
for seed in range(6):
    X, y = make_fix(n=2000, seed=seed)
    K = 4
    skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
    skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
    a_skb = auc_cols(X, y, skb_cols)
    m_ord = mrmr_order(X, y, seed)
    a_m4 = auc_cols(X, y, m_ord[:K])
    # MI frontier
    mi = mutual_info_classif(X.values, y.values, random_state=seed)
    i_ord = [X.columns[i] for i in np.argsort(mi)[::-1]]
    rm = {k: auc_cols(X, y, m_ord[:k]) for k in KS}
    ri = {k: auc_cols(X, y, i_ord[:k]) for k in KS}
    d4 = (a_m4 - a_skb) if not np.isnan(a_m4) else float("nan")
    eff_d = (rm[4] - ri[8]) if not np.isnan(rm[4]) else float("nan")
    mean_d = np.nanmean([rm[k] - ri[k] for k in KS])
    if not np.isnan(d4) and d4 >= 0.03: win4 += 1
    if not np.isnan(eff_d) and eff_d >= -0.01: eff += 1
    if not np.isnan(mean_d) and mean_d >= 0.02: dom += 1
    print(f"seed={seed} |m|={len(m_ord)} a_m4={a_m4:.4f} skb={a_skb:.4f} d4={d4:+.4f} | "
          f"m4={rm[4]:.4f} i8={ri[8]:.4f} effd={eff_d:+.4f} | meand={mean_d:+.4f}")
print(f"WIN4(>=0.03)={win4}/6  EFF(m4>=i8-0.01)={eff}/6  DOM(mean>=0.02)={dom}/6")

print("\n=== honest tie: signal_plus_noise K=4 (seeds 0-5) ===")
ties = 0
for seed in range(6):
    Xn, yn, sig = make_signal_plus_noise(n=1500, p_signal=4, p_noise=16, seed=seed)
    X, y = as_df(Xn, yn)
    K = 4
    skb = SelectKBest(mutual_info_classif, k=K).fit(X.values, y.values)
    skb_cols = [X.columns[i] for i in np.flatnonzero(skb.get_support())]
    a_skb = auc_cols(X, y, skb_cols)
    m_ord = mrmr_order(X, y, seed)
    a_m = auc_cols(X, y, m_ord)
    d = (a_m - a_skb) if not np.isnan(a_m) else float("nan")
    if not np.isnan(d) and d >= -0.02: ties += 1
    print(f"seed={seed} |m|={len(m_ord)} a_mfull={a_m:.4f} skb={a_skb:.4f} d={d:+.4f}")
print(f"TIE(mfull>=skb-0.02)={ties}/6")
