"""RFECV importance-metric shootout: impurity (auto) vs permutation vs shap, across all scenarios x seeds.

Answers 'which importance metric should be the RFECV default' per the accurate-then-fastest rule: the metric
that wins downstream honest-holdout AUC on the MAJORITY of (scenario, seed) cells, with speed as the tie-break.
Writes _results/importance_shootout.jsonl incrementally + a progress heartbeat.
"""
from __future__ import annotations
import os, sys, time, json, traceback
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
from scenarios import SCENARIOS, make

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, "importance_shootout.jsonl")
PROG = os.path.join(OUT, "importance_shootout.progress.txt")

SEEDS = [0, 1]
IMPORTANCE = ["auto", "permutation", "shap"]


def downstream(Xtr, Xte, ytr, yte, cols):
    out = {}
    for name, mk in {
        "lgbm": lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1),
        "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }.items():
        try:
            m = mk(); m.fit(Xtr[cols], ytr); out[name] = round(float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1])), 4)
        except Exception:
            out[name] = None
    return out


def log(msg):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def main():
    open(RES, "w").close(); open(PROG, "w").close()
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
    cells = [(sc, sd, ig) for sc in SCENARIOS for sd in SEEDS for ig in IMPORTANCE]
    log(f"START cells={len(cells)} scenarios={list(SCENARIOS)} seeds={SEEDS} importance={IMPORTANCE}")
    for i, (sc, sd, ig) in enumerate(cells, 1):
        X, y, t = make(sc, sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=sd, stratify=y)
        row = dict(scenario=sc, seed=sd, importance=ig)
        try:
            fi = FIConfig(importance_getter=ig, n_features_selection_rule="one_se_min")
            sgc = SearchConfig(max_refits=12, max_runtime_mins=3)
            r = RFECV(estimator=lgb.LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1),
                      cv=3, scoring=None, verbose=0, fi_config=fi, search_config=sgc, random_state=0)
            t0 = time.time(); r.fit(Xtr, ytr); row["fit_s"] = round(time.time() - t0, 1)
            cols = [c for c in r.get_feature_names_out() if c in X.columns]
            if not cols:
                cols = list(X.columns[:1])
            base = set(t["base"]); noise = set(t["noise"])
            row.update(n_feat=len(cols), base_recall=round(len(set(cols) & base) / max(1, len(base)), 3),
                       noise_sel=len(set(cols) & noise), auc=downstream(Xtr, Xte, ytr, yte, cols))
            row["auc_mean"] = round(float(np.mean([v for v in row["auc"].values() if v is not None])), 4)
            log(f"[{i}/{len(cells)}] {sc}/{sd}/{ig}: n={row['n_feat']} rec={row['base_recall']} noise={row['noise_sel']} "
                f"fit={row['fit_s']}s auc={row['auc']}")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"; row["tb"] = traceback.format_exc()[-800:]
            log(f"[{i}/{len(cells)}] {sc}/{sd}/{ig}: ERROR {row['error']}")
        with open(RES, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    log("DONE")
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(RES, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        log("no successful cells"); return
    df = pd.DataFrame(ok)
    log("\n=== mean over (scenario,seed) by importance ===")
    g = df.groupby("importance").agg(lgbm=("auc", lambda s: round(np.mean([a.get("lgbm") for a in s if a.get("lgbm")]), 4)),
                                     auc_mean=("auc_mean", lambda s: round(np.mean(s), 4)),
                                     n_feat=("n_feat", lambda s: round(np.mean(s), 1)),
                                     base_recall=("base_recall", lambda s: round(np.mean(s), 3)),
                                     noise_sel=("noise_sel", lambda s: round(np.mean(s), 1)),
                                     fit_s=("fit_s", lambda s: round(np.mean(s), 1)))
    log(g.to_string())
    # per-cell winner by lgbm AUC
    log("\n=== win counts (best lgbm AUC per scenario,seed) ===")
    wins = {ig: 0 for ig in IMPORTANCE}
    for (sc, sd), grp in df.groupby(["scenario", "seed"]):
        best = max(grp.to_dict("records"), key=lambda r: (r["auc"].get("lgbm") or -1))
        wins[best["importance"]] += 1
    log(str(wins))


if __name__ == "__main__":
    main()
