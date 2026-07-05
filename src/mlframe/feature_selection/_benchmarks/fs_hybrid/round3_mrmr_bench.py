"""Round-3 MRMR levers (all kwargs, no class edits): raise base_recall + clean the FE pair-pool.

MRMR-FE is the dominant strategy (auc 0.835) but base_recall is only ~0.62 (drops true bases) and the probe showed
it engineers SPURIOUS noise-only products (mul(log(noise_1),abs(noise_2)) etc.). Tests, vs the mrmr_fe default:
  M3-1  dcd_off / aggr_off  : is the low base_recall caused by DCD cluster pruning / cluster-aggregate replacement?
  M3-2  rescue_on           : does the run_additional_rfecv rescue re-recover dropped bases?
  M3-6  relfloor_0.02       : does a lower relevance-gain floor recover weak true bases?
  M3-3  fe_pairs_25         : richer FE pair budget?
  M3-4/7 fe_strict          : tighter synergy/engineered-MI prevalence -> fewer spurious noise products?
Measures base_recall, raw-noise selected, #spurious engineered (recipe name has no 'inf_' token), downstream AUC.
"""
from __future__ import annotations
import os, sys, time, re
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
from synth import make_dataset
from mlframe.feature_selection.filters import MRMR

SEEDS = [0, 1, 2]
_SAFE = re.compile(r"^[A-Za-z0-9_]+$")
CONFIGS = {
    "default": dict(),
    "dcd_off": dict(dcd_enable=False),
    "aggr_off": dict(cluster_aggregate_enable=False),
    "rescue_on": dict(run_additional_rfecv_minutes=0.5),
    "relfloor_0.02": dict(min_relevance_gain_relative_to_first=0.02),
    "fe_pairs_25": dict(fe_max_pair_features=25),
    "fe_strict": dict(fe_synergy_min_prevalence=1.5, fe_min_engineered_mi_prevalence=0.97),
}


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
        base, noise = set(t["base"]), set(t["noise"])
        for name, kw in CONFIGS.items():
            try:
                t0 = time.time()
                m = MRMR(verbose=0, fe_max_steps=1, n_jobs=-1, random_seed=0, **kw); m.fit(Xtr, ytr)
                dt = time.time() - t0
                out_cols = list(m.transform(Xtr.iloc[:5]).columns)
                ren, k = {}, 0
                for c in out_cols:
                    ren[c] = c if _SAFE.match(str(c)) else f"eng_{k}"; k += (0 if _SAFE.match(str(c)) else 1)
                raw_sel = [c for c in out_cols if _SAFE.match(str(c)) and c in X.columns]
                eng = [c for c in out_cols if c not in X.columns]
                # spurious engineered = recipe references NO informative column (no 'inf_' token)
                spurious = sum(1 for c in eng if "inf_" not in str(c))
                Ztr = m.transform(Xtr).copy(); Ztr.columns = [ren[c] for c in Ztr.columns]
                Zte = m.transform(Xte).copy(); Zte.columns = [ren[c] for c in Zte.columns]
                a = downstream(Ztr, Zte, ytr, yte); am = round(float(np.nanmean(list(a.values()))), 4)
                row = dict(seed=sd, config=name, n=Ztr.shape[1], n_eng=len(eng), spurious_eng=spurious,
                           base=len(set(raw_sel) & base), base_recall=round(len(set(raw_sel) & base) / len(base), 3),
                           noise=len(set(raw_sel) & noise), fit_s=round(dt, 1), auc=a, auc_mean=am)
            except Exception as e:
                row = dict(seed=sd, config=name, error=f"{type(e).__name__}: {e}")
            rows.append(row)
            print(f"sd{sd} {name:14s} " + (row.get("error") or
                  f"n={row['n']:2d} eng={row['n_eng']:2d} spur={row['spurious_eng']:2d} base={row['base']}/{len(base)} "
                  f"rec={row['base_recall']} noise={row['noise']} {row['fit_s']:5.1f}s mean={row['auc_mean']}"), flush=True)
    df = pd.DataFrame([r for r in rows if not r.get("error")])
    print("\n=== mean over seeds ===")
    print(df.groupby("config").agg(auc_mean=("auc_mean", "mean"), base_recall=("base_recall", "mean"),
                                   noise=("noise", "mean"), n_eng=("n_eng", "mean"), spurious_eng=("spurious_eng", "mean"),
                                   n=("n", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
