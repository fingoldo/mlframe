"""Round-2 RFECV R2r-2: cross-bootstrap support_ aggregation vs single-fit (variance reduction).

For each outer seed: single RFECV fit vs an AGGREGATE that frequency-votes support_ over B bootstrap resamples
of the train (keep features selected in >= half the B fits). Measures (i) downstream AUC mean + ITS cross-seed
std (the robustness target) and (ii) selection stability (Jaccard of selected sets across outer seeds).
Impurity importance (fast) - the question is about the SELECTION aggregation, not the importance metric.
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import make
from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

SCEN = ["base", "highnoise", "manyredundant"]
OUTER_SEEDS = [0, 1, 2, 3]
B = 4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")


def fit_support(Xtr, ytr, seed):
    r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=120, verbose=-1),
              cv=3, scoring=None, verbose=0,
              fi_config=FIConfig(importance_getter="feature_importances_", n_features_selection_rule="one_se_max"),
              search_config=SearchConfig(max_refits=12, max_runtime_mins=2), random_state=seed)
    r.fit(Xtr, ytr)
    return [c for c in r.get_feature_names_out() if c in Xtr.columns]


def auc(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return float("nan")
    m = lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1]))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main():
    rows = []
    for sc in SCEN:
        single_sets, agg_sets, single_aucs, agg_aucs = [], [], [], []
        for sd in OUTER_SEEDS:
            X, y, t = make(sc, sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
            Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
            # single fit
            ss = fit_support(Xtr, ytr, sd); single_sets.append(ss); single_aucs.append(auc(Xtr, Xte, ytr, yte, ss))
            # aggregate over B bootstrap resamples (frequency vote >= B/2)
            cnt = Counter()
            for b in range(B):
                rng = np.random.default_rng(1000 * sd + b)
                idx = rng.choice(len(Xtr), len(Xtr), replace=True)
                Xb, yb = Xtr.iloc[idx].reset_index(drop=True), ytr.iloc[idx].reset_index(drop=True)
                cnt.update(fit_support(Xb, yb, b))
            agg = [c for c in X.columns if cnt.get(c, 0) >= int(np.ceil(B / 2))]
            if not agg:
                agg = [c for c, _ in cnt.most_common(5)]
            agg_sets.append(agg); agg_aucs.append(auc(Xtr, Xte, ytr, yte, agg))
            print(f"{sc:14s} sd{sd} single n={len(ss)} auc={single_aucs[-1]:.4f} | agg n={len(agg)} auc={agg_aucs[-1]:.4f}", flush=True)
        # cross-seed stability (pairwise Jaccard)
        def mean_jac(sets):
            ps = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
            return float(np.mean(ps)) if ps else 1.0
        row = dict(scenario=sc,
                   single_auc_mean=round(float(np.nanmean(single_aucs)), 4), single_auc_std=round(float(np.nanstd(single_aucs)), 4),
                   agg_auc_mean=round(float(np.nanmean(agg_aucs)), 4), agg_auc_std=round(float(np.nanstd(agg_aucs)), 4),
                   single_jaccard=round(mean_jac(single_sets), 3), agg_jaccard=round(mean_jac(agg_sets), 3))
        rows.append(row)
        print(f"  -> {sc}: single auc {row['single_auc_mean']}+-{row['single_auc_std']} jac {row['single_jaccard']} | "
              f"agg auc {row['agg_auc_mean']}+-{row['agg_auc_std']} jac {row['agg_jaccard']}", flush=True)
    with open(os.path.join(OUT, "round2_rfecv_crossseed.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    print("\n=== summary ===")
    print(df.to_string(index=False))
    print(f"\nmean AUC: single {df.single_auc_mean.mean():.4f} agg {df.agg_auc_mean.mean():.4f} | "
          f"mean cross-seed AUC std: single {df.single_auc_std.mean():.4f} agg {df.agg_auc_std.mean():.4f} | "
          f"mean Jaccard: single {df.single_jaccard.mean():.3f} agg {df.agg_jaccard.mean():.3f}")


if __name__ == "__main__":
    main()
