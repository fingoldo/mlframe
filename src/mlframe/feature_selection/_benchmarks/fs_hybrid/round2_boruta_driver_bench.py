"""Round-2 BorutaShap R2b-5 (driver) + R2b-6 (cluster pre-merge).

R2b-5: is the SHAP per-trial importance driver worth its cost? The round-1 importance shootout flagged SHAP
as the worst+slowest signal. Here we run BorutaShap with importance_measure='gini' vs 'Shap' across the full
6-scenario x 2-seed bed and compare downstream honest-holdout AUC, accepted-noise, and fit time. If gini ties
or beats SHAP while being much faster, gini is the most-accurate-then-fastest default driver (CLAUDE.md S6).

R2b-6: collapse raw-correlation clusters to one representative BEFORE the shadow gate, run gini-Boruta on the
representatives, then re-expand accepted reps to their members. Targets the redundant scenarios where copies
dilute the shadow comparison and let a noise column slip past. Measured on the same bed (manyredundant is the
discriminating cell). Uses a plain greedy |corr|>=thr clustering here to test the IDEA; production would reuse
the existing DCD/SU clustering.
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import SCENARIOS, make
from mlframe.feature_selection.boruta_shap import BorutaShap

SEEDS = [0, 1]
N_TRIALS = 30
PERM_REPEATS = 2
CORR_THR = 0.92
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(OUT, exist_ok=True)


def downstream(Xtr, Xte, ytr, yte, cols):
    if not cols:
        return {"lgbm": float("nan"), "logit": float("nan"), "knn": float("nan")}
    o = {}
    o["lgbm"] = roc_auc_score(yte, lgb.LGBMClassifier(n_estimators=300, verbose=-1).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["logit"] = roc_auc_score(yte, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    o["knn"] = roc_auc_score(yte, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)).fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1])
    return {k: round(float(v), 4) for k, v in o.items()}


def boruta(X, y, measure, train_or_test="train", perm_repeats=3):
    b = BorutaShap(model=RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=0),
                   importance_measure=measure, permutation_n_repeats=perm_repeats, classification=True,
                   n_trials=N_TRIALS, percentile=95, train_or_test=train_or_test, verbose=False, random_state=0)
    b.fit(X, y)
    return [c for c in b.selected_features_ if c in X.columns]


def corr_clusters(X, thr=CORR_THR):
    """Greedy |corr|>=thr clustering -> (representatives, rep_of: rep -> member list incl. rep)."""
    cols = list(X.columns)
    C = np.corrcoef(X.values, rowvar=False)
    C = np.nan_to_num(C)
    reps, rep_of, assigned = [], {}, set()
    for i, c in enumerate(cols):
        if c in assigned:
            continue
        reps.append(c); rep_of[c] = [c]; assigned.add(c)
        for j in range(i + 1, len(cols)):
            if cols[j] not in assigned and abs(C[i, j]) >= thr:
                rep_of[c].append(cols[j]); assigned.add(cols[j])
    return reps, rep_of


def boruta_premerge(X, y):
    reps, rep_of = corr_clusters(X)
    sel_reps = boruta(X[reps], y, "gini")
    expanded = []
    for r in sel_reps:
        expanded.extend(rep_of.get(r, [r]))
    return [c for c in dict.fromkeys(expanded) if c in X.columns], len(reps)


def main():
    rows = []
    for sc in SCENARIOS:
        for sd in SEEDS:
            X, y, t = make(sc, sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
            noise, base = set(t["noise"]), set(t["base"])
            methods = {}
            t0 = time.time(); methods["gini"] = (boruta(Xtr, ytr, "gini"), time.time() - t0, "")
            t0 = time.time(); methods["shap"] = (boruta(Xtr, ytr, "Shap"), time.time() - t0, "")
            # permutation in its HONEST held-out mode (train_or_test="test"); a 1-cell smoke showed held-out recovers
            # 7/7 base vs in-bag 6/7, so the in-bag variant is dropped here to keep the full-bed run tractable.
            t0 = time.time(); methods["perm_test"] = (boruta(Xtr, ytr, "permutation", train_or_test="test", perm_repeats=PERM_REPEATS), time.time() - t0, "held-out")
            t0 = time.time(); cols, nreps = boruta_premerge(Xtr, ytr); methods["premerge_gini"] = (cols, time.time() - t0, f"reps={nreps}")
            for name, (cols, dt, extra) in methods.items():
                cols = [c for c in cols if c in X.columns]
                a = downstream(Xtr, Xte, ytr, yte, cols)
                am = round(float(np.nanmean(list(a.values()))), 4)
                row = dict(scenario=sc, seed=sd, method=name, n=len(cols), base=len(set(cols) & base),
                           noise=len(set(cols) & noise), fit_s=round(dt, 1), auc=a, auc_mean=am, extra=extra)
                rows.append(row)
                print(f"{sc:14s} sd{sd} {name:14s} n={row['n']:2d} base={row['base']}/{len(base)} noise={row['noise']:2d} {dt:6.1f}s auc={a} mean={am} {extra}", flush=True)
    with open(os.path.join(OUT, "round2_boruta_driver.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    print("\n=== mean over scenarios x seeds ===")
    print(df.groupby("method").agg(auc_mean=("auc_mean", "mean"), lgbm=("auc", lambda s: np.mean([a["lgbm"] for a in s])),
                                   noise=("noise", "mean"), n=("n", "mean"), fit_s=("fit_s", "mean")).round(4).to_string())
    wins = {}
    for (sc, sd), g in df.groupby(["scenario", "seed"]):
        b = g.loc[g["auc_mean"].idxmax(), "method"]; wins[b] = wins.get(b, 0) + 1
    print("win counts (best auc_mean):", wins)
    # R2b-5 driver head-to-head: gini vs shap vs permutation (held-out + in-bag), ignoring premerge
    drivers = ["gini", "shap", "perm_test"]
    g2 = df[df.method.isin(drivers)]
    gv = {m: round(s.auc_mean.mean(), 4) for m, s in g2.groupby("method")}
    sv = {m: round(s.fit_s.mean(), 1) for m, s in g2.groupby("method")}
    dwins = {}
    for (sc, sd), g in g2.groupby(["scenario", "seed"]):
        w = g.loc[g.auc_mean.idxmax(), "method"]; dwins[w] = dwins.get(w, 0) + 1
    print(f"R2b-5 drivers mean auc {gv}")
    print(f"        driver fit_s  {sv}")
    print(f"        driver wins   {dwins} (of {len(SCENARIOS)*len(SEEDS)} cells)")


if __name__ == "__main__":
    main()
