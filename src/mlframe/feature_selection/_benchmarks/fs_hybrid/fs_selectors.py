"""Uniform adapters over the four mlframe selectors + cascade/ensemble combinators.

Every adapter exposes .fit(X,y) and .transform(X) returning a pandas DataFrame with
ASCII-safe column names (RFECV mis-handles numpy input; LightGBM rejects names with
parentheses/commas, which MRMR feature-engineering produces). Cascades chain fitted
adapters; ensembles union/intersect two raw selected sets.
"""
from __future__ import annotations
import os, re, time
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import pandas as pd

_SAFE = re.compile(r"^[A-Za-z0-9_]+$")


def _is_safe(name: str) -> bool:
    return bool(_SAFE.match(str(name)))


# ----------------------------------------------------------------------------- base adapters
class AllSel:
    name = "all"
    def fit(self, X, y):
        self.cols_ = list(X.columns); self.raw_selected_ = list(X.columns); self.n_engineered_ = 0; return self
    def transform(self, X):
        return X[self.cols_]


class MRMRSel:
    """MRMR filter (fe=False) or MRMR with feature-engineering (fe=True)."""
    def __init__(self, fe: bool):
        self.fe = fe; self.name = "mrmr_fe" if fe else "mrmr_filter"
    def fit(self, X, y):
        from mlframe.feature_selection.filters import MRMR
        self.m_ = MRMR(verbose=0, fe_max_steps=(1 if self.fe else 0), n_jobs=-1, random_seed=0)
        self.m_.fit(X, y)
        out = self.m_.transform(X.iloc[:5])  # peek output column names
        self.out_cols_ = list(out.columns)
        self.rename_ = {}
        k = 0
        for c in self.out_cols_:
            if _is_safe(c):
                self.rename_[c] = c
            else:
                self.rename_[c] = f"eng_{k}"; k += 1
        self.n_engineered_ = k
        self.raw_selected_ = [c for c in self.out_cols_ if _is_safe(c) and c in X.columns]
        return self
    def transform(self, X):
        df = self.m_.transform(X)
        df = df.copy(); df.columns = [self.rename_[c] for c in df.columns]
        return df


class BorutaSel:
    def __init__(self, stability_subsamples: int = 0):
        self.stability_subsamples = stability_subsamples
        self.name = "boruta_stable" if stability_subsamples else "boruta"
    def fit(self, X, y):
        from mlframe.feature_selection.boruta_shap import BorutaShap
        from sklearn.ensemble import RandomForestClassifier
        self.b_ = BorutaShap(
            model=RandomForestClassifier(n_estimators=80, max_depth=None, n_jobs=-1, random_state=0),
            importance_measure="gini", classification=True, n_trials=60, percentile=95,
            pvalue=0.05, verbose=False, random_state=0,
            stability_subsamples=self.stability_subsamples,
            stability_subsample_fraction=0.75, stability_threshold=1.0,  # intersection: reliably drops draw-level spurious
        )
        self.b_.fit(X, y)
        self.raw_selected_ = [c for c in self.b_.selected_features_ if c in X.columns]
        if not self.raw_selected_:
            self.raw_selected_ = list(X.columns[:1])
        self.n_engineered_ = 0
        return self
    def transform(self, X):
        return X[self.raw_selected_]


class RFECVSel:
    def __init__(self, kind: str):
        self.kind = kind; self.name = f"rfecv_{kind}"
    def _estimator(self):
        if self.kind == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1)
        elif self.kind == "logit":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, C=1.0)
        raise ValueError(self.kind)
    def fit(self, X, y):
        from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
        fi = FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min")
        sc = SearchConfig(max_refits=18, max_runtime_mins=3)
        self.r_ = RFECV(estimator=self._estimator(), cv=3, scoring=None, verbose=0, fi_config=fi, search_config=sc, random_state=0)
        self.r_.fit(X, y)
        self.raw_selected_ = [c for c in self.r_.get_feature_names_out() if c in X.columns]
        if not self.raw_selected_:
            self.raw_selected_ = list(X.columns[:1])
        self.n_engineered_ = sum(1 for c in self.r_.get_feature_names_out() if c not in X.columns)
        # engineered columns from upstream FE survive as their (renamed-safe) names:
        self.all_selected_ = list(self.r_.get_feature_names_out())
        return self
    def transform(self, X):
        keep = [c for c in self.all_selected_ if c in X.columns]
        return X[keep]


class ShapSel:
    name = "shap_proxied"
    def fit(self, X, y):
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
        p = X.shape[1]
        # within_cluster_refine=False: its default honest-loss proxy over-prunes real signal that a strong
        # downstream uses (measured: 6 feats / LGBM 0.73 vs 18 feats / 0.77 with refine off). See ShapProxiedFS docstring.
        self.s_ = ShapProxiedFS(
            classification=True, n_splits=3, top_n=20, min_features=8,
            prefilter_top=min(40, p), prefilter_n_estimators=60,
            oof_shap_n_estimators=60, revalidation_n_estimators=60,
            n_revalidation_models=2, trust_guard=True, trust_guard_n_estimators=20,
            cluster_features="auto", within_cluster_refine=False, random_state=0, verbose=False,
        )
        self.s_.fit(X, y)
        self.raw_selected_ = [c for c in self.s_.selected_features_ if c in X.columns]
        if not self.raw_selected_:
            self.raw_selected_ = list(X.columns[:1])
        self.all_selected_ = list(self.s_.selected_features_)
        self.n_engineered_ = sum(1 for c in self.all_selected_ if c not in X.columns)
        try:
            self.report_ = {k: self.s_.shap_proxy_report_.get(k) for k in ("trust_guard", "cluster")}
        except Exception:
            self.report_ = {}
        return self
    def transform(self, X):
        keep = [c for c in self.all_selected_ if c in X.columns]
        return X[keep]


# ----------------------------------------------------------------------------- combinators
class Cascade:
    """Chain fitted selectors: each stage fits on the previous stage's output."""
    def __init__(self, name, *stages):
        self.name = name; self.stages = stages
    def fit(self, X, y):
        cur = X
        for st in self.stages:
            st.fit(cur, y); cur = st.transform(cur)
        last = self.stages[-1]
        self.raw_selected_ = [c for c in cur.columns if c in X.columns]
        self.n_engineered_ = sum(1 for c in cur.columns if c not in X.columns)
        self.final_cols_ = list(cur.columns)
        return self
    def transform(self, X):
        cur = X
        for st in self.stages:
            cur = st.transform(cur)
        return cur


class Ensemble:
    """Union or intersection of two selectors' RAW selected sets (model-agnostic merge)."""
    def __init__(self, name, a, b, op: str):
        self.name = name; self.a = a; self.b = b; self.op = op
    def fit(self, X, y):
        self.a.fit(X, y); self.b.fit(X, y)
        sa, sb = set(self.a.raw_selected_), set(self.b.raw_selected_)
        merged = (sa | sb) if self.op == "union" else (sa & sb)
        if not merged:
            merged = sa or sb or set(list(X.columns)[:1])
        self.raw_selected_ = [c for c in X.columns if c in merged]
        self.n_engineered_ = 0
        return self
    def transform(self, X):
        return X[self.raw_selected_]
