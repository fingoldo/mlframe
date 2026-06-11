"""Definitive: make_signal_plus_noise honest tie via downstream_auc (transform incl
engineered combo), FE-default config. p_signal=3 (matches the working additive case;
4 additive signals may not collapse to a single FE combo cleanly)."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from feature_selection._biz_val_synth import make_signal_plus_noise, as_df, downstream_auc


def fit(X, y, seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(random_seed=seed, verbose=0, cv=3, run_additional_rfecv_minutes=False,
                    min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1).fit(X, y)


def skb_auc(X, y, k):
    cols = [X.columns[i] for i in np.flatnonzero(
        SelectKBest(mutual_info_classif, k=k).fit(X.values, y.values).get_support())]
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=5, scoring="roc_auc").mean())


for psig in (3, 4):
    print(f"=== make_signal_plus_noise p_signal={psig}, downstream_auc(transform) vs SKB ===")
    ties = 0
    for seed in range(6):
        Xn, yn, sig = make_signal_plus_noise(n=1500, p_signal=psig, p_noise=16, seed=seed)
        X, y = as_df(Xn, yn)
        a_skb = skb_auc(X, y, psig)
        sel = fit(X, y, seed)
        a_m = downstream_auc(sel, X, y)
        nout = sel.transform(X).shape[1]
        d = a_m - a_skb if not np.isnan(a_m) else float("nan")
        if not np.isnan(a_m) and a_m >= a_skb - 0.02:
            ties += 1
        names = list(sel.get_feature_names_out())[:3]
        print(f" seed={seed} skb={a_skb:.4f} mrmr_ds={a_m:.4f} n_out={nout} d={d:+.4f} names={names}")
    print(f" TIE(mrmr_ds>=skb-0.02)={ties}/6\n")
