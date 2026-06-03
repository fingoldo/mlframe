"""Round-3 fix #2: does the hybrid earn its cost on a HARDER bed where signal is split so no single selector wins?

Strategies: mrmr_fe (FE, misses weak-sparse), rfecv_lgbm_perm (weak-sparse, misses FE), boruta_fe, H2
(mrmr_fe->rfecv_logit), hybrid (anchor, 3 members + FE), hybrid_rfecv (anchor + an RFECV 4th member -- tests whether
the weak-sparse block revives H3-4 on hard data). Per-block recovery (weak-sparse, strong) + downstream AUC on
hard_synth.make_hard_dataset. If the hybrid beats BOTH mrmr_fe and rfecv, the composition is justified on hard data.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_synth import make_hard_dataset
from hybrid_selector import HybridSelector
import fs_selectors as S

SEEDS = [0, 1]


class HybridRFECV(HybridSelector):
    """hybrid + an RFECV 4th member on X_aug (tests H3-4 on the hard bed's weak-sparse block)."""
    def fit(self, X, y):
        super().fit(X, y)
        try:
            from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
            r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
                      cv=3, scoring=None, verbose=0, fi_config=FIConfig(importance_getter="permutation", n_features_selection_rule="one_se_min"),
                      search_config=SearchConfig(max_refits=12, max_runtime_mins=3), random_state=self.random_state)
            r.fit(self._Xaug_[self.relevant_], self._y_)
            self.member_selections_["rfecv"] = [c for c in r.get_feature_names_out() if c in self._Xaug_.columns]
            sel = self._combine(self.member_selections_, list(self._Xaug_.columns))
            eng = set(self._eng_rename.values())
            self.raw_selected_ = [c for c in self._Xaug_.columns if c in set(sel)] or list(self._Xaug_.columns[:1])
            self.n_engineered_ = sum(1 for c in self.raw_selected_ if c in eng)
        except Exception as e:
            warnings.warn(f"HybridRFECV degraded ({type(e).__name__}: {e})")
        return self


def downstream(Ztr, Zte, ytr, yte):
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def build(name):
    return {"mrmr_fe": lambda: S.MRMRSel(fe=True),
            "rfecv_perm": lambda: S.RFECVSel("lgbm_perm"),
            "boruta_fe": lambda: S.Cascade("boruta_fe", S.MRMRSel(fe=True), S.BorutaSel()),
            "H2_mrmrfe_rfecvlogit": lambda: S.Cascade("H2", S.MRMRSel(fe=True), S.RFECVSel("logit")),
            "hybrid": lambda: HybridSelector(vote=1, use_fe=True, anchor_fe=True),
            "hybrid_noanchor": lambda: HybridSelector(vote=1, use_fe=True, anchor_fe=False),
            "hybrid_rfecv": lambda: HybridRFECV(vote=1, use_fe=True, anchor_fe=True)}[name]()


def main():
    rows = []
    for sd in SEEDS:
        X, y, t = make_hard_dataset(n_samples=5000, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        strong, weak = set(t["strong"]), set(t["weak_sparse"])
        for name in ("mrmr_fe", "rfecv_perm", "boruta_fe", "H2_mrmrfe_rfecvlogit", "hybrid", "hybrid_noanchor", "hybrid_rfecv"):
            t0 = time.time(); sel = build(name); sel.fit(Xtr, ytr); dt = time.time() - t0
            Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
            raw = set(c for c in getattr(sel, "raw_selected_", []) if c in X.columns)
            a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
            rows.append(dict(seed=sd, strategy=name, n=int(Ztr.shape[1]), n_eng=int(getattr(sel, "n_engineered_", 0)),
                             strong=len(raw & strong), weak=len(raw & weak), fit_s=round(dt, 1), auc=a, auc_mean=am))
            print(f"sd{sd} {name:20s} n={rows[-1]['n']:3d} eng={rows[-1]['n_eng']:2d} strong={rows[-1]['strong']}/4 "
                  f"weak={rows[-1]['weak']}/8 {dt:6.1f}s mean={am} auc={a}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== mean over seeds ===")
    g = df.groupby("strategy").agg(auc_mean=("auc_mean", "mean"), strong=("strong", "mean"), weak=("weak", "mean"),
                                   n=("n", "mean"), n_eng=("n_eng", "mean"), fit_s=("fit_s", "mean"),
                                   lgbm=("auc", lambda s: round(np.mean([a["lgbm"] for a in s]), 4)),
                                   logit=("auc", lambda s: round(np.mean([a["logit"] for a in s]), 4))).round(4)
    print(g.sort_values("auc_mean", ascending=False).to_string())


if __name__ == "__main__":
    main()
