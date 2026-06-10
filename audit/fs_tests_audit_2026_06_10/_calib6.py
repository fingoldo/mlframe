"""Diagnose the signal_plus_noise collapse + test the SELECTOR_SPECS factory config
and a few alternatives. Goal: a config that (1) keeps signal on plain linear data
(honest tie) AND (2) diversifies on the dominant-cluster fixture (redundancy win)."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from feature_selection._biz_val_synth import make_signal_plus_noise, as_df
from feature_selection._selector_factories import SELECTOR_SPECS


def auc_cols(X, y, cols, cv=5):
    cols = list(cols)
    if not cols:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def order_of(sel, X):
    names = list(X.columns)
    return [names[i] for i in np.asarray(sel.support_, dtype=int)]


def fit_factory(X, y, seed):
    """The SELECTOR_SPECS MRMR factory config (random_seed overridden per seed)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False,
                   full_npermutations=3, random_seed=seed, min_features_fallback=1,
                   verbose=False).fit(X, y)
    return order_of(sel, X)


def fit_cfg(X, y, seed, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(random_seed=seed, verbose=0, cv=3,
                   run_additional_rfecv_minutes=False, **kw).fit(X, y)
    return order_of(sel, X)


print("=== signal_plus_noise: which config KEEPS the 4 signals? ===")
for seed in range(4):
    Xn, yn, sig = make_signal_plus_noise(n=1500, p_signal=4, p_noise=16, seed=seed)
    X, y = as_df(Xn, yn)
    a_skb = auc_cols(X, y, [X.columns[i] for i in np.flatnonzero(
        SelectKBest(mutual_info_classif, k=4).fit(X.values, y.values).get_support())])
    o_fac = fit_factory(X, y, seed)
    o_npm0 = fit_cfg(X, y, seed, min_relevance_gain=0.0, full_npermutations=0, min_features_fallback=4)
    o_simple = fit_cfg(X, y, seed, min_relevance_gain=0.0, full_npermutations=0,
                       use_simple_mode=True, min_features_fallback=4)
    print(f" seed={seed} skb={a_skb:.4f}")
    print(f"   factory  |{len(o_fac)}| auc_full={auc_cols(X,y,o_fac):.4f} {o_fac[:6]}")
    print(f"   npm0     |{len(o_npm0)}| auc_full={auc_cols(X,y,o_npm0):.4f} {o_npm0[:6]}")
    print(f"   simple0  |{len(o_simple)}| auc_full={auc_cols(X,y,o_simple):.4f} {o_simple[:6]}")
