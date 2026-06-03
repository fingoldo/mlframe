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

    def __init__(self, vote: int = 1, prescreen: bool = True, expand_clusters: bool = False,
                 fi_guard: bool = False, corr_thr: float = 0.92, use_mrmr: bool = True,
                 use_fe: bool = True, fe_max_steps: int = 1, boruta_driver: str = "gini", anchor_fe: bool = True,
                 random_state: int = 0, name: str = "hybrid"):
        self.vote = vote
        self.prescreen = prescreen
        self.expand_clusters = expand_clusters
        self.fi_guard = fi_guard
        self.corr_thr = corr_thr
        self.use_mrmr = use_mrmr
        # use_fe: let the MRMR member feature-ENGINEER (fe_max_steps) and SHARE its engineered columns to every other
        # member via a once-built augmented frame X_aug=[raw|engineered]. This closes the measured ~0.05 AUC gap to
        # mrmr_fe: the bed's pure-interaction signal (inf_4*inf_5, ~0 marginal) is unrecoverable by selection alone;
        # only FE creates the term. Engineering is replayed leakage-free at transform time via the fitted MRMR member.
        self.use_fe = use_fe
        self.fe_max_steps = fe_max_steps
        # boruta_driver: importance measure for the BorutaShap member. "gini" is the default. The held-out
        # "permutation" driver was tried as a noise-leak fix (gini admits ~1-3 spurious columns that, under vote=1,
        # cap the linear downstream): MEASURED net-negative without FE (it only partially cut noise 2.33->1.33 AND
        # cost recall 0.96->0.88 -> auc 0.784->0.779, at 2.5x runtime) and merely TIED gini with FE on (0.8299 vs
        # 0.8288) at ~2x cost. The real cure for the noise-capped linear downstream is feature engineering (use_fe),
        # which lifts logit ~0.75->0.85 so residual noise no longer matters. So gini stays the default; permutation
        # remains available as the high-precision option.
        self.boruta_driver = boruta_driver
        # anchor_fe: when FE is on, ANCHOR the final selection on the MRMR substrate -- keep ALL of MRMR's picks
        # (raw + engineered) verbatim, then ADD only the raw clusters the OTHER members confirm that MRMR missed.
        # By construction selected ⊇ mrmr_selected, so the hybrid can never score below its strongest FE-aware member
        # (the measured defect: the plain cluster-vote could re-emit a raw operand over MRMR's engineered term and so
        # TRAIL mrmr_fe). The other members thus only ADD complementary recall, never subtract the FE signal.
        self.anchor_fe = anchor_fe
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
        self._mrmr_member, self._eng_names, self._eng_rename = None, [], {}
        try:
            from mlframe.feature_selection.filters import MRMR
            # fe_strict (round-3 M3-4/M3-7): tighter synergy + engineered-MI prevalence gates than MRMR's defaults
            # (1.15 / 0.90). Measured on make_dataset: cuts spurious noise-product engineered features (~1.0 -> 0.67)
            # and HALVES the engineered set (15 -> 7) while raising standalone mrmr_fe AUC 0.831 -> 0.837 -- a cleaner,
            # more parsimonious FE substrate to share to the other members.
            fe_strict = dict(fe_synergy_min_prevalence=1.5, fe_min_engineered_mi_prevalence=0.97) if self.use_fe else {}
            m = MRMR(verbose=0, fe_max_steps=(self.fe_max_steps if self.use_fe else 0), n_jobs=-1,
                     random_seed=self.random_state, retain_artifacts=True, retain_bins=True, **fe_strict)
            m.fit(X, y)
            selected = [c for c in m.get_feature_names_out() if c in X.columns]
            try:
                artifacts = m.export_artifacts()
            except Exception:
                artifacts = None
            # Capture the engineered columns (MRMR outputs not in X) so they can be SHARED to the other members via
            # the augmented frame; map their non-ASCII recipe names to LightGBM-safe eng_N names (stable order).
            if self.use_fe:
                try:
                    out_cols = list(m.transform(X.iloc[:5]).columns)
                    k = 0
                    for c in out_cols:
                        if c not in X.columns:
                            self._eng_names.append(c); self._eng_rename[c] = f"eng_{k}"; k += 1
                    self._mrmr_member = m
                except Exception:
                    self._mrmr_member, self._eng_names, self._eng_rename = None, [], {}
            return selected, artifacts
        except Exception as e:
            warnings.warn(f"HybridSelector: MRMR stage degraded ({type(e).__name__}: {e})")
            return [], None

    def _augment(self, X):
        """Return [raw X | MRMR engineered columns (safe-renamed)]. Engineering is replayed leakage-free by the
        fitted MRMR member (pure function of X, no y). Identity when FE is off or produced no engineered columns."""
        if not self._eng_names or self._mrmr_member is None:
            return X
        try:
            out = self._mrmr_member.transform(X)
            eng = out[self._eng_names].copy()
            eng.columns = [self._eng_rename[c] for c in self._eng_names]
            base = X.reset_index(drop=True)
            eng = eng.reset_index(drop=True)
            aug = pd.concat([base, eng], axis=1)
            return aug.loc[:, ~aug.columns.duplicated()]
        except Exception as e:
            warnings.warn(f"HybridSelector: FE augment failed ({type(e).__name__}: {e}); using raw X")
            return X

    # ------------------------------------------------------------------ reused members on the shared narrowed space
    def _run_shap(self, X, y, relevant, artifacts):
        """ShapProxiedFS fed the SHARED artifacts via precomputed= (restricted to the relevant survivor set)."""
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
        precomputed = None
        # MRMR artifacts cover only RAW columns; if the relevant set carries engineered cols (use_fe), ShapProxiedFS
        # would warn + discard the precomputed entirely (feature_names mismatch). Only share artifacts when every
        # relevant column is covered (the pure-selection case) -- otherwise let the shap member recompute cleanly.
        if artifacts is not None:
            try:
                from mlframe.feature_selection._shap_proxy_precomputed import restrict_artifacts
                names = list(artifacts.get("feature_names", []))
                if relevant and all(c in names for c in relevant):
                    precomputed = restrict_artifacts(artifacts, [names.index(c) for c in relevant])
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
        """BorutaShap on the SHARED cluster representatives (premerge, R2b-6), then re-expand accepted reps. Uses the
        boruta_driver importance measure ("permutation" held-out by default -> ~0 noise leak, vs "gini" which admits
        the top spurious column and caps the linear downstream under vote=1)."""
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
        driver = str(self.boruta_driver).lower()
        # held-out permutation needs the 30% split (train_or_test="test"); gini works in-bag.
        tot = "test" if driver == "permutation" else "train"
        b = BorutaShap(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=self.random_state),
                       importance_measure=self.boruta_driver, permutation_n_repeats=2, classification=True,
                       n_trials=50, percentile=95, train_or_test=tot, verbose=False, random_state=self.random_state)
        b.fit(X[reps], y)
        accepted_reps = [c for c in b.selected_features_ if c in reps]
        expanded = []
        for r in accepted_reps:
            expanded.extend(rep_members.get(r, [r]))
        return [c for c in dict.fromkeys(expanded) if c in X.columns]

    # ------------------------------------------------------------------ fit / transform
    def fit(self, X, y):
        # STAGE 0 -- MRMR FIRST (it engineers the shared FE columns), then build the augmented frame X_aug once;
        # every downstream shared artifact + member then operates on raw+engineered features.
        self.mrmr_selected_, self.artifacts_ = (self._run_mrmr(X, y) if self.use_mrmr else ([], None))
        if not self.use_mrmr:
            self._mrmr_member, self._eng_names, self._eng_rename = None, [], {}
        X_aug = self._augment(X)
        cols = list(X_aug.columns)
        engineered = set(self._eng_rename.values())
        self._Xaug_, self._y_ = X_aug, y  # stashed for combine-rule variants/diagnostics (benchmark composition class)

        # shared artifacts, computed ONCE on the augmented frame (FI honest for engineered cols too)
        self.fi_ = self._shared_perm_fi(X_aug, y)
        self.reps_, self.members_ = corr_clusters(X_aug, self.corr_thr)
        self.cluster_of_ = {f: r for r, ms in self.members_.items() for f in ms}

        # STAGE 1 -- shared relevance pre-screen (held-out permutation FI > 0, OR MRMR-relevant, OR engineered:
        # engineered columns already passed MRMR's FE gate, so they are never dropped by the raw-FI prescreen)
        if self.prescreen:
            mrmr_set = set(self.mrmr_selected_)
            relevant = [c for c in cols if self.fi_.get(c, 0.0) > 0.0 or c in mrmr_set or c in engineered]
        else:
            relevant = list(cols)
        if len(relevant) < 2:
            relevant = list(cols)
        self.relevant_ = relevant

        # STAGE 2 -- reuse the member selectors on the shared, narrowed augmented space
        member_sel = {}
        member_sel["mrmr"] = [c for c in (self.mrmr_selected_ + sorted(engineered)) if c in relevant] or list(relevant)
        try:
            member_sel["shap"] = self._run_shap(X_aug, y, relevant, self.artifacts_)
        except Exception as e:
            warnings.warn(f"HybridSelector: shap member degraded ({type(e).__name__}: {e})")
            member_sel["shap"] = []
        try:
            member_sel["boruta"] = self._run_boruta_premerge(X_aug, y, relevant)
        except Exception as e:
            warnings.warn(f"HybridSelector: boruta member degraded ({type(e).__name__}: {e})")
            member_sel["boruta"] = []
        self.member_selections_ = member_sel

        # STAGE 3 -- cluster-aware vote over the shared clusters (pure, deterministic; extracted for unit testing)
        selected = self._combine(member_sel, cols)
        self.raw_selected_ = [c for c in cols if c in set(selected)] or cols[:1]
        self.n_engineered_ = sum(1 for c in self.raw_selected_ if c in engineered)
        return self

    def _combine(self, member_sel, cols):
        """Cluster-aware vote: a cluster is kept when >= ``vote`` members picked any of its features; each kept
        cluster contributes its highest-shared-FI member (or all members under ``expand_clusters``). Pure function
        of the shared state (self.fi_ / self.members_ / self.cluster_of_) and the per-member selections, so it is
        unit-testable without the expensive member fits."""
        if self.anchor_fe and member_sel.get("mrmr"):
            return self._combine_anchored(member_sel, cols)
        cluster_votes = defaultdict(set)
        for m, sel in member_sel.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    cluster_votes[r].add(m)
        support = {r: len(voters) for r, voters in cluster_votes.items()}

        def _rep_fi(r):
            return max((self.fi_.get(m, 0.0) for m in self.members_[r] if m in cols), default=0.0)

        # FI-CREDIBILITY GUARD (off by default -- MEASURED net loss): a cluster confirmed by only ONE member is
        # admitted only if its shared-permutation-FI clears the credibility floor = the median rep-FI of the
        # CONSENSUS clusters (>= 2 members). The intent was to drop the single-member noise leak while keeping
        # real single-member features. Benched (round2_hybrid_bench.py, 3 seeds on make_dataset): it DOES cut
        # noise 2.33 -> 0.33, but the single-member BASE features are exactly the weaker-FI ones, so they fall
        # below the consensus-FI median too -> base_recall collapses 0.958 -> 0.792 and auc_mean drops 0.784 ->
        # 0.772 (it degenerates to the vote=2 consensus set). The leaked noise's AUC cost is small (lgbm tolerates
        # it) while the lost recall is not. So fi_guard stays OFF by default; keep it only when a downstream needs
        # a cleaner, lower-recall set. Uses the already-shared FI, no extra compute.
        consensus_fis = [_rep_fi(r) for r, s in support.items() if s >= 2]
        floor = float(np.median(consensus_fis)) if (self.fi_guard and consensus_fis) else float("-inf")
        chosen = []
        for r, s in support.items():
            if s >= max(self.vote, 2):           # consensus -> always keep
                chosen.append(r)
            elif s >= self.vote and _rep_fi(r) >= floor:   # single-member -> must clear the credibility floor
                chosen.append(r)
        if not chosen:  # guard/vote too strict for this dataset -> fall back to any-member union of clusters
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
        return selected

    def _combine_anchored(self, member_sel, cols):
        """Anchor on the MRMR/FE substrate (see anchor_fe). Keep ALL of MRMR's picks verbatim (raw + engineered),
        then ADD the clusters MRMR missed that >= ``vote`` of the OTHER members (shap/boruta) confirm. selected is a
        superset of mrmr_selected, so AUC >= mrmr_fe by construction; the other members only add complementary recall."""
        mrmr_sel = [c for c in member_sel.get("mrmr", []) if c in cols]
        selected = list(dict.fromkeys(mrmr_sel))                 # FE substrate kept verbatim (engineered preserved)
        anchored_clusters = {self.cluster_of_.get(c) for c in mrmr_sel}
        others = [m for m in member_sel if m != "mrmr"]
        votes = defaultdict(set)
        for m in others:
            for f in member_sel[m]:
                r = self.cluster_of_.get(f)
                if r is not None and r not in anchored_clusters:
                    votes[r].add(m)
        for r, voters in votes.items():
            if len(voters) >= self.vote:                          # add only clusters the other members confirm
                ms = [m for m in self.members_[r] if m in cols]
                if not ms:
                    continue
                if self.expand_clusters:
                    selected.extend(ms)
                else:
                    selected.append(max(ms, key=lambda f: self.fi_.get(f, 0.0)))
        return list(dict.fromkeys(selected))

    def transform(self, X):
        # replay FE so engineered selections (eng_N) are available, then slice to the selected set
        X_aug = self._augment(X)
        keep = [c for c in self.raw_selected_ if c in X_aug.columns]
        return X_aug[keep] if keep else X_aug.iloc[:, :1]
