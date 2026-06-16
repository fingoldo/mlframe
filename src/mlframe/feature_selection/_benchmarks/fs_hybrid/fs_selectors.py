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
        # fe_strict (round-3 M3-4/M3-7): tighter FE prevalence gates -> standalone mrmr_fe auc 0.831->0.837, half the
        # engineered features, fewer spurious noise-products. Applied only when feature-engineering is on.
        fe_kw = dict(fe_synergy_min_prevalence=1.5, fe_min_engineered_mi_prevalence=0.97) if self.fe else {}
        self.m_ = MRMR(verbose=0, fe_max_steps=(1 if self.fe else 0), n_jobs=-1, random_seed=0, **fe_kw)
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
    def __init__(self, kind: str, survivor_fe: bool = False):
        # kind "lgbm_perm": LightGBM estimator but OOF-permutation importance for elimination instead of impurity;
        # measured +0.029 mean lgbm AUC over impurity across 3 seeds (positive every seed), at ~4-5x fit cost.
        # survivor_fe (round-3 R3-1): after selection, CV-gate pairwise products/squares of the top survivors and
        # append the ones that lift 3-fold CV AUC > 1 SE. Recovers a pure-interaction term when BOTH operands survive
        # elimination (measured +0.015 mean AUC; forms the true inf_a*inf_b). Partial: marginal elimination can drop
        # one operand first, so it cannot match pre-elimination FE (mrmr_fe). Replays the products at transform.
        self.kind = kind; self.survivor_fe = survivor_fe
        self.name = f"rfecv_{kind}" + ("_fe" if survivor_fe else "")
        self._importance_getter = "permutation" if kind.endswith("_perm") else "auto"
    def _estimator(self):
        base = self.kind[:-5] if self.kind.endswith("_perm") else self.kind
        if base == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1)
        elif base == "logit":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, C=1.0)
        raise ValueError(self.kind)
    def _survivor_products(self, X, y, survivors, k=8):
        """CV-gated pairwise products/squares of the top-k survivors (by LightGBM impurity); keep a product only if
        adding it raises 3-fold CV ROC-AUC over the survivor set by > 1 SE. Returns list of (a, b) operand pairs."""
        import itertools, lightgbm as lgb
        from sklearn.model_selection import cross_val_score
        if len(survivors) < 2:
            return []
        m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(X[survivors], y)
        top = [c for _, c in sorted(zip(m.feature_importances_, survivors), reverse=True)][:k]
        base = cross_val_score(lgb.LGBMClassifier(n_estimators=120, verbose=-1), X[survivors], y, cv=3, scoring="roc_auc")
        thr = base.mean() + base.std() / (len(base) ** 0.5)
        kept = []
        for a, b in list(itertools.combinations(top, 2)) + [(c, c) for c in top]:
            aug = pd.concat([X[survivors], (X[a] * X[b]).rename("prod")], axis=1)
            sc = cross_val_score(lgb.LGBMClassifier(n_estimators=120, verbose=-1), aug, y, cv=3, scoring="roc_auc")
            if sc.mean() > thr:
                kept.append((a, b))
        return kept
    def _apply_products(self, X):
        out = X[[c for c in self.all_selected_ if c in X.columns]].copy()
        for a, b in getattr(self, "products_", []):
            if a in X.columns and b in X.columns:
                out[f"prod__{a}__{b}"] = X[a] * X[b]
        return out
    def fit(self, X, y):
        from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
        fi = FIConfig(importance_getter=self._importance_getter, n_features_selection_rule="one_se_min")
        sc = SearchConfig(max_refits=18, max_runtime_mins=3)
        self.r_ = RFECV(estimator=self._estimator(), cv=3, scoring=None, verbose=0, fi_config=fi, search_config=sc, random_state=0)
        self.r_.fit(X, y)
        self.raw_selected_ = [c for c in self.r_.get_feature_names_out() if c in X.columns]
        if not self.raw_selected_:
            self.raw_selected_ = list(X.columns[:1])
        self.all_selected_ = list(self.r_.get_feature_names_out())
        self.products_ = self._survivor_products(X, y, self.raw_selected_) if self.survivor_fe else []
        self.n_engineered_ = sum(1 for c in self.all_selected_ if c not in X.columns) + len(self.products_)
        return self
    def transform(self, X):
        return self._apply_products(X)


class ShapSel:
    name = "shap_proxied"
    def __init__(self, su_seeded_interactions: bool = False):
        # su_seeded_interactions (A4-4): flip the IN-CLASS opt-in SNR-gated sparse-interaction path.
        # When True the selector runs its OWN pairwise-SU synergy screen + SNR gate internally (no
        # externally-seeded product columns), so the bench can compare the prototype seeding against
        # the production in-class path. Default False preserves the legacy additive selector.
        self.su_seeded_interactions = bool(su_seeded_interactions)
    def fit(self, X, y):
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
        p = X.shape[1]
        # within_cluster_refine=True + parsimony_tol=0.005: the measured optimum on this bed (beats refine=False in
        # 4/6 cells, +~0.6pt downstream LGBM AUC, ~half the features). The old refine=False workaround was for the
        # loose 0.02 tol, now fixed at the class default. See ShapProxiedFS docstring for the calibration.
        self.s_ = ShapProxiedFS(
            classification=True, n_splits=3, top_n=20, min_features=8,
            prefilter_top=min(40, p), prefilter_n_estimators=60,
            oof_shap_n_estimators=60, revalidation_n_estimators=60,
            n_revalidation_models=2, trust_guard=True, trust_guard_n_estimators=20,
            cluster_features="auto", within_cluster_refine=True, parsimony_tol=0.005,
            su_seeded_interactions=self.su_seeded_interactions,
            random_state=0, verbose=False,
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


class ReliefFSel:
    """ReliefF ranker (skrebate) as a selector. ReliefF scores features by local
    near-hit/near-miss separation; we keep features with POSITIVE weight (irrelevant
    features get <=0 weight under Relief), an adaptive, parameter-light cut. Falls back
    to top-max(8, sqrt(F)) if none are positive. Pure ranker -> no engineered columns.

    Rows are subsampled to <=RELIEFF_ROW_CAP for the fit only (ReliefF is O(n^2*F));
    the selected column set is then applied to the full frame. n_neighbors small for speed.
    """
    name = "relieff"
    RELIEFF_ROW_CAP = 2000

    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        from skrebate import ReliefF
        Xn = X.to_numpy(dtype=np.float64) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float64)
        yn = np.asarray(y).astype(np.int64)
        n = Xn.shape[0]
        if n > self.RELIEFF_ROW_CAP:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, self.RELIEFF_ROW_CAP, replace=False)
            Xn, yn = Xn[idx], yn[idx]
        k = min(self.n_neighbors, max(1, (yn == np.bincount(yn).argmin()).sum() - 1))
        r = ReliefF(n_neighbors=k, n_jobs=-1)
        r.fit(Xn, yn)
        w = np.asarray(r.feature_importances_, dtype=np.float64)
        cols = list(X.columns)
        keep = [cols[i] for i in range(len(cols)) if w[i] > 0]
        if not keep:
            kk = max(8, int(np.sqrt(len(cols))))
            keep = [cols[i] for i in np.argsort(-w)[:kk]]
        self.raw_selected_ = keep
        self.n_engineered_ = 0
        return self

    def transform(self, X):
        return X[self.raw_selected_]


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
