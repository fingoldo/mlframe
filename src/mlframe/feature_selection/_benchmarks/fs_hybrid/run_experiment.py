"""Feature-selection hybrid experiment.

For each (seed, strategy): fit selector on TRAIN, then train 3 downstream model
families (LightGBM / Logistic / kNN) on the selected (possibly engineered) features
and score honest AUC on a held-out TEST set the selector never saw. Also record
ground-truth recovery (vs known causal/redundant/noise blocks), parsimony, cost.

Writes results.jsonl incrementally and progress.txt checkpoints (one line per cell).
"""
from __future__ import annotations
import os, sys, time, json, traceback
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

try:  # runnable both as ``python -m ...fs_hybrid.run_experiment`` and as a plain script
    from .synth import make_dataset
    from . import fs_selectors as S
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from synth import make_dataset
    import fs_selectors as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(OUT, "results.jsonl")
PROGRESS = os.path.join(OUT, "progress.txt")

CORE_SEEDS = [0, 1, 2]
SHAP_SEEDS = [0]


def downstream_models():
    return {
        "lgbm": lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, n_jobs=-1, verbose=-1),
        "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }


def build_roster():
    """name -> (factory(), seeds). factory builds a fresh unfitted adapter."""
    R = {}
    R["all"] = (lambda: S.AllSel(), CORE_SEEDS)
    R["mrmr_filter"] = (lambda: S.MRMRSel(fe=False), CORE_SEEDS)
    R["mrmr_fe"] = (lambda: S.MRMRSel(fe=True), CORE_SEEDS)
    R["boruta"] = (lambda: S.BorutaSel(), CORE_SEEDS)
    R["boruta_stable"] = (lambda: S.BorutaSel(stability_subsamples=10), CORE_SEEDS)
    R["rfecv_lgbm"] = (lambda: S.RFECVSel("lgbm"), CORE_SEEDS)
    R["rfecv_logit"] = (lambda: S.RFECVSel("logit"), CORE_SEEDS)
    # hybrids
    R["H1_mrmrfilter__rfecv_lgbm"] = (lambda: S.Cascade("H1", S.MRMRSel(fe=False), S.RFECVSel("lgbm")), CORE_SEEDS)
    R["H2_mrmrfe__rfecv_logit"] = (lambda: S.Cascade("H2", S.MRMRSel(fe=True), S.RFECVSel("logit")), CORE_SEEDS)
    R["H3_boruta__rfecv_lgbm"] = (lambda: S.Cascade("H3", S.BorutaSel(), S.RFECVSel("lgbm")), CORE_SEEDS)
    R["H_union_mrmr_boruta"] = (lambda: S.Ensemble("Hu", S.MRMRSel(fe=False), S.BorutaSel(), "union"), CORE_SEEDS)
    R["H_intersect_mrmr_boruta"] = (lambda: S.Ensemble("Hi", S.MRMRSel(fe=False), S.BorutaSel(), "intersect"), CORE_SEEDS)
    R["H5_mrmr_boruta__rfecv_lgbm"] = (lambda: S.Cascade("H5", S.MRMRSel(fe=False), S.BorutaSel(), S.RFECVSel("lgbm")), CORE_SEEDS)
    R["H7_mrmr_borutastable__rfecv_lgbm"] = (lambda: S.Cascade("H7", S.MRMRSel(fe=False), S.BorutaSel(stability_subsamples=10), S.RFECVSel("lgbm")), CORE_SEEDS)
    # shap-proxied (cost-limited)
    R["shap_proxied"] = (lambda: S.ShapSel(), SHAP_SEEDS)
    R["H4_mrmrfilter__shap"] = (lambda: S.Cascade("H4", S.MRMRSel(fe=False), S.ShapSel()), SHAP_SEEDS)
    R["H6_mrmrfe__shap"] = (lambda: S.Cascade("H6", S.MRMRSel(fe=True), S.ShapSel()), SHAP_SEEDS)
    return R


def recovery(raw_selected, truth):
    base = set(truth["base"]); noise = set(truth["noise"]); red = set(truth["relevant"]) - base
    sel = set(raw_selected)
    tp = len(sel & base); fn = len(base - sel)
    n_noise = len(sel & noise); n_red = len(sel & red)
    prec = tp / max(1, len(sel & (base | noise | red)))
    rec = tp / max(1, len(base))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(base_recall=round(rec, 3), base_found=tp, base_missed=fn,
                n_noise_selected=n_noise, n_redundant_selected=n_red,
                precision_on_base=round(prec, 3), f1=round(f1, 3))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    open(RESULTS, "w").close(); open(PROGRESS, "w").close()
    roster = build_roster()
    models = downstream_models()
    total = sum(len(seeds) for _, seeds in roster.values())
    log(f"START total_cells={total} strategies={len(roster)} seeds_core={CORE_SEEDS}")
    cell = 0
    for name, (factory, seeds) in roster.items():
        for seed in seeds:
            cell += 1
            X, y, truth = make_dataset(n_samples=5000, seed=seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
            row = {"strategy": name, "seed": seed}
            try:
                sel = factory()
                t0 = time.time()
                sel.fit(Xtr, ytr)
                row["fit_seconds"] = round(time.time() - t0, 2)
                Ztr = sel.transform(Xtr); Zte = sel.transform(Xte)
                row["n_features"] = int(Ztr.shape[1])
                row["n_engineered"] = int(getattr(sel, "n_engineered_", 0))
                row["raw_selected"] = list(getattr(sel, "raw_selected_", []))
                row.update(recovery(getattr(sel, "raw_selected_", []), truth))
                aucs = {}
                for mname, mfac in models.items():
                    try:
                        clf = mfac(); clf.fit(Ztr, ytr)
                        aucs[mname] = round(float(roc_auc_score(yte, clf.predict_proba(Zte)[:, 1])), 4)
                    except Exception as e:
                        aucs[mname] = None; row[f"err_{mname}"] = f"{type(e).__name__}: {e}"
                row["auc"] = aucs
                row["auc_mean"] = round(float(np.mean([v for v in aucs.values() if v is not None])), 4) if any(aucs.values()) else None
                log(f"[{cell}/{total}] {name} seed={seed} n={row['n_features']} eng={row['n_engineered']} "
                    f"fit={row['fit_seconds']}s rec={row.get('base_recall')} noise={row.get('n_noise_selected')} auc={aucs}")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback"] = traceback.format_exc()[-1500:]
                log(f"[{cell}/{total}] {name} seed={seed} ERROR {type(e).__name__}: {e}")
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    log("DONE")


if __name__ == "__main__":
    main()
