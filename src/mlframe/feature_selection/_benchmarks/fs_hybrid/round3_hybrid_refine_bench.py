"""Round-3 hybrid combine-rule refinements H3-2..H3-6 -- MEASURED (not deferred), vs the default FE-hybrid.

Each refinement is a HybridSelector subclass overriding only the combine/member logic, so the change is isolated:
  H3-2 vote_oofauc   : weight each member's cluster vote by its selected-subset 3-fold OOF AUC (not equal count).
  H3-3 fe_protect    : in a mixed cluster, emit the ENGINEERED term (not the raw operand) regardless of FI.
  H3-4 rfecv_member  : add an RFECV member (the best pure-selection AUC selector) to the vote on X_aug.
  H3-6 decouple      : a cluster is confirmed by ANY raw vote, but emits its engineered member if present.
  H3-5 auto_combine  : pick (vote,expand) by internal 3-fold OOF AUC over the candidate configs.
Reports downstream honest-holdout AUC vs the default hybrid; ship any that beats it by > cross-seed noise.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import make_dataset
from hybrid_selector import HybridSelector

SEEDS = [0, 1, 2]


def _oof_auc(X, y, cols):
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 1:
        return 0.5
    return float(cross_val_score(lgb.LGBMClassifier(n_estimators=120, verbose=-1), X[cols], y, cv=3, scoring="roc_auc").mean())


class H32(HybridSelector):  # vote weighted by each member's subset OOF-AUC
    def _combine(self, member_sel, cols):
        Xa, y = self._Xaug_, self._y_
        w = {m: max(0.0, _oof_auc(Xa, y, sel) - 0.5) for m, sel in member_sel.items()}
        cl_w = defaultdict(float)
        for m, sel in member_sel.items():
            for f in set(self.cluster_of_.get(c) for c in sel):
                if f is not None:
                    cl_w[f] += w.get(m, 0.0)
        tot = sum(w.values()) or 1.0
        chosen = [r for r, wt in cl_w.items() if wt / tot >= 0.33]  # >= a third of the total skill-weight
        if not chosen:
            chosen = list(cl_w)
        out = []
        for r in chosen:
            ms = [m for m in self.members_[r] if m in cols]
            if ms:
                out.append(max(ms, key=lambda f: self.fi_.get(f, 0.0)))
        return out


class H33(HybridSelector):  # emit engineered term in a mixed cluster (FE-protect)
    def _combine(self, member_sel, cols):
        eng = set(self._eng_rename.values())
        base_sel = super()._combine(member_sel, cols)
        # for each selected feature, if its cluster contains an engineered member, swap to it
        out = []
        for c in base_sel:
            r = self.cluster_of_.get(c)
            members = [m for m in self.members_.get(r, [c]) if m in cols] if r else [c]
            eng_members = [m for m in members if m in eng]
            out.append(eng_members[0] if eng_members else c)
        return list(dict.fromkeys(out))


class H34(HybridSelector):  # add RFECV as a 4th member
    def _run_rfecv(self, X, y, relevant):
        from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
        r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
                  cv=3, scoring=None, verbose=0,
                  fi_config=FIConfig(importance_getter="feature_importances_", n_features_selection_rule="one_se_min"),
                  search_config=SearchConfig(max_refits=12, max_runtime_mins=2), random_state=self.random_state)
        r.fit(X[relevant], y)
        return [c for c in r.get_feature_names_out() if c in X.columns]

    def fit(self, X, y):
        super().fit(X, y)
        try:
            rf = self._run_rfecv(self._Xaug_, self._y_, self.relevant_)
            self.member_selections_["rfecv"] = rf
            selected = self._combine(self.member_selections_, list(self._Xaug_.columns))
            self.raw_selected_ = [c for c in self._Xaug_.columns if c in set(selected)] or list(self._Xaug_.columns[:1])
            self.n_engineered_ = sum(1 for c in self.raw_selected_ if c in set(self._eng_rename.values()))
        except Exception as e:
            warnings.warn(f"H34 rfecv member degraded ({type(e).__name__}: {e})")
        return self


class H35(HybridSelector):  # auto-pick (vote, expand) by internal OOF-AUC
    def fit(self, X, y):
        super().fit(X, y)
        Xa, y_ = self._Xaug_, self._y_
        best, best_auc = (self.vote, self.expand_clusters), -1.0
        for vote in (1, 2):
            for expand in (False, True):
                self.vote, self.expand_clusters = vote, expand
                sel = self._combine(self.member_selections_, list(Xa.columns))
                au = _oof_auc(Xa, y_, sel)
                if au > best_auc:
                    best_auc, best = au, (vote, expand)
        self.vote, self.expand_clusters = best
        selected = self._combine(self.member_selections_, list(Xa.columns))
        self.raw_selected_ = [c for c in Xa.columns if c in set(selected)] or list(Xa.columns[:1])
        self.n_engineered_ = sum(1 for c in self.raw_selected_ if c in set(self._eng_rename.values()))
        return self


class H36(HybridSelector):  # cluster confirmed by any vote; emit engineered member if present (decouple)
    def _combine(self, member_sel, cols):
        eng = set(self._eng_rename.values())
        votes = defaultdict(set)
        for m, sel in member_sel.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    votes[r].add(m)
        chosen = [r for r, v in votes.items() if len(v) >= self.vote] or list(votes)
        out = []
        for r in chosen:
            ms = [m for m in self.members_[r] if m in cols]
            if not ms:
                continue
            eng_ms = [m for m in ms if m in eng]
            out.append(eng_ms[0] if eng_ms else max(ms, key=lambda f: self.fi_.get(f, 0.0)))
        return out


VARIANTS = {"default": HybridSelector, "H32_voteOOF": H32, "H33_feprotect": H33,
            "H34_rfecv": H34, "H35_autocombine": H35, "H36_decouple": H36}


def downstream(Ztr, Zte, ytr, yte):
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        base = set(t["base"])
        for name, cls in VARIANTS.items():
            try:
                t0 = time.time(); h = cls(vote=1, use_fe=True, random_state=0); h.fit(Xtr, ytr); dt = time.time() - t0
                Ztr, Zte = h.transform(Xtr), h.transform(Xte)
                a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
                raw = [c for c in h.raw_selected_ if c in X.columns]
                rows.append(dict(seed=sd, variant=name, n=Ztr.shape[1], base_recall=round(len(set(raw) & base) / len(base), 3),
                                 fit_s=round(dt, 1), auc_mean=am))
                print(f"sd{sd} {name:16s} n={Ztr.shape[1]:2d} rec={rows[-1]['base_recall']} {dt:6.1f}s mean={am} auc={a}", flush=True)
            except Exception as e:
                print(f"sd{sd} {name:16s} ERROR {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds (vs default) ===")
    g = df.groupby("variant").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                  n=("n", "mean"), fit_s=("fit_s", "mean")).round(4)
    d = g.loc["default", "auc_mean"] if "default" in g.index else float("nan")
    g["delta_vs_default"] = (g["auc_mean"] - d).round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
