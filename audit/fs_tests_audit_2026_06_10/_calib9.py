"""Does FE reproduce the documented add()-combo tie on additive signal?
Test the EXACT additive target from mrmr.py docstring: y=sign(x0+x1+x2) with FE on.
Also confirm transform output names to see if engineered features appear."""
import sys, warnings, numpy as np, pandas as pd
sys.path.insert(0, "tests")
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def auc_cols(X, y, cols, cv=5):
    cols = list(cols)
    if not cols:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=400), X[cols], y, cv=cv, scoring="roc_auc").mean())


def fit(X, y, seed, **kw):
    from mlframe.feature_selection.filters.mrmr import MRMR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(random_seed=seed, verbose=0, cv=3, run_additional_rfecv_minutes=False, **kw).fit(X, y)


# exact docstring target: 3 additive signals + noise, n=2000
print("=== additive y=sign(x0+x1+x2) FE-on (docstring contract) ===")
for seed in range(4):
    rng = np.random.default_rng(seed)
    n = 2000
    Xs = rng.standard_normal((n, 3))
    Xn = rng.standard_normal((n, 12))
    Xa = np.column_stack([Xs, Xn])
    cols = [f"x{i}" for i in range(15)]
    X = pd.DataFrame(Xa, columns=cols)
    y = pd.Series((Xs.sum(1) + 0.3*rng.standard_normal(n) > 0).astype(np.int64))
    a_skb = auc_cols(X, y, [X.columns[i] for i in np.flatnonzero(
        SelectKBest(mutual_info_classif, k=3).fit(X.values, y.values).get_support())])
    for tag, kw in [("fe_default", dict(min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1)),
                    ("fe_steps2", dict(min_relevance_gain=0.0, full_npermutations=3, min_features_fallback=1,
                                       fe_max_steps=2))]:
        try:
            sel = fit(X, y, seed, **kw)
            names_out = list(sel.get_feature_names_out())
            Xt = sel.transform(X)
            a = float(cross_val_score(LogisticRegression(max_iter=400), Xt, y, cv=5,
                                      scoring="roc_auc").mean()) if Xt.shape[1] else float("nan")
        except Exception as e:
            names_out = [f"ERR:{e}"]; a = float("nan")
        print(f" seed={seed} skb={a_skb:.4f} [{tag}] n_out={len(names_out)} auc={a:.4f} names={names_out[:8]}")
