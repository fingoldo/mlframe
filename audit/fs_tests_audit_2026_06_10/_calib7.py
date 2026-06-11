"""(a) Does FE-ON let MRMR TIE SelectKBest on signal_plus_noise via downstream_auc
    (engineered add() combo)?  (b) A cleaner dominant-cluster fixture for a reliable
    diversification win, comparing MRMR-FULL vs SelectKBest at MRMR's own support
    size (apples-to-apples K)."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from feature_selection._biz_val_synth import make_signal_plus_noise, as_df, downstream_auc


def auc_cols(X, y, cols, cv=5):
    cols = list(cols)
    if not cols:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def fit_fe(X, y, seed):
    """FE-ON MRMR (default fe path) -- builds engineered add()/combos."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(random_seed=seed, verbose=0, cv=3, run_additional_rfecv_minutes=False,
                   min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1).fit(X, y)
    return sel


print("=== (a) signal_plus_noise: FE-ON MRMR downstream_auc (transform incl engineered) vs SKB ===")
for seed in range(4):
    Xn, yn, sig = make_signal_plus_noise(n=1500, p_signal=4, p_noise=16, seed=seed)
    X, y = as_df(Xn, yn)
    a_skb = auc_cols(X, y, [X.columns[i] for i in np.flatnonzero(
        SelectKBest(mutual_info_classif, k=4).fit(X.values, y.values).get_support())])
    sel = fit_fe(X, y, seed)
    try:
        a_fe = downstream_auc(sel, X, y)
    except Exception as e:
        a_fe = float("nan"); print("   downstream_auc err:", e)
    nfeat = sel.transform(X).shape[1]
    print(f" seed={seed} skb={a_skb:.4f} mrmr_fe_downstream={a_fe:.4f} n_out={nfeat} d={a_fe-a_skb:+.4f}")


# (b) cleaner dominant-cluster: cluster0 = strongest via WEIGHT, 4 copies each
def make_fix2(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    zc = [rng.standard_normal(n) for _ in range(3)]
    cols = {}
    for c in range(3):
        for k in range(5):  # 5 copies each
            cols[f"clu{c}_m{k}"] = zc[c] + 0.15 * rng.standard_normal(n)
    i0 = rng.standard_normal(n); i1 = rng.standard_normal(n)
    cols["indep0"] = i0; cols["indep1"] = i1
    for j in range(20):
        cols[f"noise{j}"] = rng.standard_normal(n)
    # cluster0 strongest marginal weight => MI ranks its 5 copies top-5
    score = 2.5*zc[0] + 1.2*zc[1] + 1.2*zc[2] + 1.0*i0 + 1.0*i1 + 0.3*rng.standard_normal(n)
    y = (score > np.median(score)).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y)


def fit_div(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(random_seed=seed, verbose=0, cv=3, run_additional_rfecv_minutes=False,
                   use_simple_mode=False, dcd_enable=False, min_relevance_gain=0.0,
                   min_relevance_gain_relative_to_first=0.0, max_consec_unconfirmed=60,
                   min_features_fallback=4, full_npermutations=5).fit(X, y)
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


print("\n=== (b) cleaner dominant-cluster: MRMR-FULL vs SKB@same-K (seeds 0-5) ===")
wins = 0
for seed in range(6):
    X, y = make_fix2(n=2000, seed=seed)
    order = fit_div(X, y, seed)
    Km = len(order)
    if Km == 0:
        print(f" seed={seed} MRMR EMPTY"); continue
    skb_cols = [X.columns[i] for i in np.flatnonzero(
        SelectKBest(mutual_info_classif, k=Km).fit(X.values, y.values).get_support())]
    a_m = auc_cols(X, y, order)
    a_s = auc_cols(X, y, skb_cols)
    clcov = len({c.split('_')[0] for c in order if c.startswith('clu')})
    skb_clu0 = sum(1 for c in skb_cols if c.startswith('clu0_'))
    d = a_m - a_s
    if d >= 0.03: wins += 1
    print(f" seed={seed} Km={Km} clcov={clcov} skb_clu0={skb_clu0}/{Km} "
          f"a_mrmr={a_m:.4f} a_skb={a_s:.4f} d={d:+.4f}")
print(f"WIN(mrmr_full>=skb@K +0.03)={wins}/6")
