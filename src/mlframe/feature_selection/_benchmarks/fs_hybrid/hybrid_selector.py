"""HybridSelector: a compute-once / share-many composition of the four mlframe selectors.

Design principles (both explicit in this class):
  1. COMPUTE-ONCE-SHARE-MANY. Three artifacts are computed a single time on the full X and then handed to the
     reused component selectors instead of each one recomputing them:
       - MI / SU / per-column bins: produced by MRMR (retain_artifacts=True) and injected into ShapProxiedFS via
         its ``precomputed=`` hook (skips ShapProxiedFS's own univariate pre-screen + re-binning).
       - permutation FEATURE IMPORTANCE: one honest held-out pass (mirrors RFECV's permutation importance). Drives
         the cheap relevance pre-screen that narrows X before the expensive wrappers, and breaks ties when emitting
         one representative per redundant cluster.
       - raw-correlation CLUSTERS: computed once; used both to pre-merge correlated columns before the BorutaShap
         gate (the measured R2b-6 win) and to de-duplicate the final selection.
  2. REUSE, DON'T REIMPLEMENT. The members are the real production selectors composed as whole objects
     (MRMR, ShapProxiedFS, BorutaShap); the hybrid only adds the shared-artifact plumbing and the combine rule.

Final selection is a CLUSTER-AWARE VOTE: a cluster is kept when >= ``vote`` members selected any of its features;
each kept cluster contributes its highest-permutation-FI member (parsimony), or all members when expand_clusters.

Adapter protocol matches fs_selectors.py: ``fit(X, y)`` / ``transform(X)`` / ``raw_selected_`` / ``n_engineered_``.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd


def corr_clusters(X: pd.DataFrame, thr: float = 0.92):
    """Greedy |Pearson| >= thr clustering. Returns (reps, members) where reps is the representative list (first
    column of each cluster) and members maps rep -> [member columns incl. the rep]. Computed once and shared."""
    cols = list(X.columns)
    C = np.nan_to_num(np.corrcoef(X.values, rowvar=False))
    if C.ndim == 0:  # single column
        return [cols[0]], {cols[0]: [cols[0]]}
    reps, members, assigned = [], {}, set()
    for i, c in enumerate(cols):
        if c in assigned:
            continue
        reps.append(c); members[c] = [c]; assigned.add(c)
        for j in range(i + 1, len(cols)):
            if cols[j] not in assigned and abs(C[i, j]) >= thr:
                members[c].append(cols[j]); assigned.add(cols[j])
    return reps, members


class HybridSelector:
    """Compute-once / share-many hybrid feature selector (see module docstring)."""

    def __init__(self, vote: int = 2, prescreen: bool = True, expand_clusters: bool = False,
                 corr_thr: float = 0.92, use_mrmr: bool = True, random_state: int = 0, name: str = "hybrid"):
        self.vote = vote
        self.prescreen = prescreen
        self.expand_clusters = expand_clusters
        self.corr_thr = corr_thr
        self.use_mrmr = use_mrmr
        self.random_state = random_state
        self.name = name

    # ------------------------------------------------------------------ shared-once artifacts
    def _shared_perm_fi(self, X, y):
        """One honest held-out permutation-importance pass (the shared FI). Mirrors RFECV's permutation driver:
        fit a single LightGBM on a train split, permute on the held-out split, return col -> mean importance."""
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance
        import lightgbm as lgb
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=self.random_state, stratify=y)
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        pi = permutation_importance(m, Xva, yva, n_repeats=4, random_state=self.random_state, n_jobs=-1)
        return {c: float(v) for c, v in zip(X.columns, pi.importances_mean)}

    def _run_mrmr(self, X, y):
        """MRMR as a whole object -> (selected raw columns, artifact dict). Guarded: MRMR is shared infra under
        active development, so any failure degrades to (relevance-by-FI only, no precomputed) rather than crashing."""
        try:
            from mlframe.feature_selection.filters import MRMR
            m = MRMR(verbose=0, fe_max_steps=0, n_jobs=-1, random_seed=self.random_state,
                     retain_artifacts=True, retain_bins=True)
            m.fit(X, y)
            selected = [c for c in m.get_feature_names_out() if c in X.columns]
            try:
                artifacts = m.export_artifacts()
            except Exception:
                artifacts = None
            return selected, artifacts
        except Exception as e:
            warnings.warn(f"HybridSelector: MRMR stage degraded ({type(e).__name__}: {e})")
            return [], None

    # ------------------------------------------------------------------ reused members on the shared narrowed space
    def _run_shap(self, X, y, relevant, artifacts):
        """ShapProxiedFS fed the SHARED artifacts via precomputed= (restricted to the relevant survivor set)."""
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
        precomputed = None
        if artifacts is not None:
            try:
                from mlframe.feature_selection._shap_proxy_precomputed import restrict_artifacts
                names = list(artifacts.get("feature_names", []))
                keep = [names.index(c) for c in relevant if c in names]
                if keep:
                    precomputed = restrict_artifacts(artifacts, keep)
            except Exception:
                precomputed = None
        p = len(relevant)
        s = ShapProxiedFS(classification=True, n_splits=3, top_n=20, min_features=min(8, p),
                          prefilter_top=min(40, p), prefilter_n_estimators=60, oof_shap_n_estimators=60,
                          revalidation_n_estimators=60, n_revalidation_models=2, trust_guard=True,
                          trust_guard_n_estimators=20, cluster_features="auto", within_cluster_refine=True,
                          parsimony_tol=0.005, precomputed=precomputed, random_state=self.random_state, verbose=False)
        s.fit(X[relevant], y)
        return [c for c in s.selected_features_ if c in X.columns]

    def _run_boruta_premerge(self, X, y, relevant):
        """BorutaShap(gini) on the SHARED cluster representatives (premerge, R2b-6), then re-expand accepted reps."""
        from mlframe.feature_selection.boruta_shap import BorutaShap
        from sklearn.ensemble import RandomForestClassifier
        # restrict the shared clusters to the relevant survivors; rep = highest-FI member of each restricted cluster
        rep_members = {}
        for r, ms in self.members_.items():
            keep = [m for m in ms if m in relevant]
            if keep:
                rep = max(keep, key=lambda f: self.fi_.get(f, 0.0))
                rep_members[rep] = keep
        reps = list(rep_members.keys())
        if len(reps) < 2:
            reps = list(relevant)
            rep_members = {c: [c] for c in reps}
        b = BorutaShap(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=self.random_state),
                       importance_measure="gini", classification=True, n_trials=50, percentile=95,
                       verbose=False, random_state=self.random_state)
        b.fit(X[reps], y)
        accepted_reps = [c for c in b.selected_features_ if c in reps]
        expanded = []
        for r in accepted_reps:
            expanded.extend(rep_members.get(r, [r]))
        return [c for c in dict.fromkeys(expanded) if c in X.columns]

    # ------------------------------------------------------------------ fit / transform
    def fit(self, X, y):
        cols = list(X.columns)
        # STAGE 0 -- shared artifacts, computed ONCE
        self.fi_ = self._shared_perm_fi(X, y)
        self.reps_, self.members_ = corr_clusters(X, self.corr_thr)
        self.cluster_of_ = {f: r for r, ms in self.members_.items() for f in ms}
        self.mrmr_selected_, self.artifacts_ = (self._run_mrmr(X, y) if self.use_mrmr else ([], None))

        # STAGE 1 -- shared relevance pre-screen (held-out permutation FI > 0, OR MRMR-relevant)
        if self.prescreen:
            mrmr_set = set(self.mrmr_selected_)
            relevant = [c for c in cols if self.fi_.get(c, 0.0) > 0.0 or c in mrmr_set]
        else:
            relevant = list(cols)
        if len(relevant) < 2:
            relevant = list(cols)
        self.relevant_ = relevant

        # STAGE 2 -- reuse the member selectors on the shared, narrowed space
        member_sel = {}
        member_sel["mrmr"] = [c for c in self.mrmr_selected_ if c in relevant] or list(relevant)
        try:
            member_sel["shap"] = self._run_shap(X, y, relevant, self.artifacts_)
        except Exception as e:
            warnings.warn(f"HybridSelector: shap member degraded ({type(e).__name__}: {e})")
            member_sel["shap"] = []
        try:
            member_sel["boruta"] = self._run_boruta_premerge(X, y, relevant)
        except Exception as e:
            warnings.warn(f"HybridSelector: boruta member degraded ({type(e).__name__}: {e})")
            member_sel["boruta"] = []
        self.member_selections_ = member_sel

        # STAGE 3 -- cluster-aware vote: a cluster is kept when >= vote members picked any of its features
        cluster_votes = defaultdict(set)
        for m, sel in member_sel.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    cluster_votes[r].add(m)
        chosen = [r for r, voters in cluster_votes.items() if len(voters) >= self.vote]
        if not chosen:  # vote too strict for this dataset -> fall back to any-member union of clusters
            chosen = list(cluster_votes.keys())
        selected = []
        for r in chosen:
            ms = [m for m in self.members_[r] if m in cols]
            if not ms:
                continue
            if self.expand_clusters:
                selected.extend(ms)
            else:
                selected.append(max(ms, key=lambda f: self.fi_.get(f, 0.0)))
        self.raw_selected_ = [c for c in cols if c in set(selected)] or cols[:1]
        self.n_engineered_ = 0
        return self

    def transform(self, X):
        return X[self.raw_selected_]
