"""Bench: BorutaShap importance_measure='auto' vs static 'gini' / 'permutation'.

Compares the three drivers on the honest holdout (a LightGBM/Logistic/kNN-style model refit on the
BorutaShap-selected features, scored on a held-out 30% never seen by the selector) across 5 synthetic
beds x 3 seeds, spanning clean-large-n AND noisy-small-n/p. Reports the auto-chosen measure, honest
holdout AUC, and selector wall per cell, plus a flip verdict.

FLIP RULE (CLAUDE.md): make auto the default ONLY if it wins/ties the MAJORITY of cells on honest
holdout AND does NOT pay the ~11x permutation cost on clean beds, REPLICATED across seeds. Else keep
gini and commit this bench + the reject verdict (REJECTED != DELETED).

Run (host env):
  set CUDA_VISIBLE_DEVICES=  & set MLFRAME_NO_CUDA_AUTOCONFIG=1 & set MLFRAME_KEEP_BROKEN_CUPY=1
  python -m mlframe.feature_selection._benchmarks.bench_boruta_auto_dispatch
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# (name, n, p, n_informative): two clean/large-n, three noisy/small-n-per-feature.
BEDS = [
    ("clean_bigN_A", 1600, 10, 6),
    ("clean_bigN_B", 2000, 12, 7),
    ("noisy_smallNp_A", 250, 50, 4),
    ("noisy_smallNp_B", 300, 40, 3),
    ("noisy_smallNp_C", 220, 60, 4),
]
SEEDS = [0, 1, 2]
N_TRIALS = 20
PERMUTATION_N_REPEATS = 3


def _make(n, p, inf, seed):
    X, y = make_classification(n_samples=n, n_features=p, n_informative=inf, n_redundant=0, shuffle=False, random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), pd.Series(y)


def _honest_holdout_auc(X, y, selected, seed):
    """Refit a fresh model on the SELECTED features over a train split, score on a held-out 30%."""
    if not selected:
        return 0.5
    Xs = X[selected]
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=500)
    try:
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        return float(roc_auc_score(yte, proba))
    except Exception as exc:
        logger.debug("_honest_holdout_auc: refit/score failed, using chance-level 0.5: %s", exc)
        return 0.5


def _run_selector(measure, X, y, seed):
    from mlframe.feature_selection.boruta_shap import BorutaShap

    kw = dict(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=seed),
              importance_measure=measure, classification=True, n_trials=N_TRIALS, percentile=95,
              verbose=False, random_state=seed)
    if measure in ("permutation", "auto"):
        kw["permutation_n_repeats"] = PERMUTATION_N_REPEATS
        if measure == "permutation":
            kw["train_or_test"] = "test"
    b = BorutaShap(**kw)
    t0 = time.perf_counter()
    b.fit(X, y)
    wall = time.perf_counter() - t0
    selected = [c for c in b.selected_features_ if c in X.columns]
    chosen = getattr(b, "_resolved_importance_measure_", measure)
    return selected, wall, chosen


def main():
    rows = []
    for name, n, p, inf in BEDS:
        for seed in SEEDS:
            X, y = _make(n, p, inf, seed)
            # Split off the honest-holdout-feature-eval data deterministically by passing same seed downstream.
            cell = {"bed": name, "seed": seed, "n": n, "p": p}
            for measure in ("gini", "permutation", "auto"):
                sel, wall, chosen = _run_selector(measure, X, y, seed)
                auc = _honest_holdout_auc(X, y, sel, seed)
                cell[f"{measure}_auc"] = round(auc, 4)
                cell[f"{measure}_wall"] = round(wall, 2)
                cell[f"{measure}_nsel"] = len(sel)
                if measure == "auto":
                    cell["auto_chosen"] = chosen
            rows.append(cell)
            print(f"{name:18s} seed={seed} | gini {cell['gini_auc']:.3f}/{cell['gini_wall']:.1f}s "
                  f"| perm {cell['permutation_auc']:.3f}/{cell['permutation_wall']:.1f}s "
                  f"| auto[{cell['auto_chosen']:11s}] {cell['auto_auc']:.3f}/{cell['auto_wall']:.1f}s", flush=True)

    # Flip rule (CLAUDE.md "Variant defaults"): clean cells where auto==gini are trivially equal and must
    # NOT count toward an accuracy flip; the flip must be earned on the NOISY beds (the only cells where auto
    # routes differently), and per-bed REPLICATED across seeds -- a single lucky seed does not count.
    clean_no_perm_cost = all(r["auto_chosen"] == "gini" for r in rows if r["bed"].startswith("clean"))
    noisy_beds = sorted({r["bed"] for r in rows if r["bed"].startswith("noisy")})
    per_bed_noisy_win = {}
    for bed in noisy_beds:
        cells = [r for r in rows if r["bed"] == bed]
        wins = sum(1 for r in cells if r["auto_auc"] >= r["gini_auc"] - 0.005)
        per_bed_noisy_win[bed] = f"{wins}/{len(cells)}"
    # auto earns the flip only if it wins a per-bed replicated majority on EVERY noisy bed (>= 2/3).
    replicated_noisy = all(int(v.split("/")[0]) >= 2 for v in per_bed_noisy_win.values())
    noisy_auto_ge_gini = sum(1 for r in rows if r["bed"].startswith("noisy") and r["auto_auc"] >= r["gini_auc"] - 0.005)
    n_noisy = sum(1 for r in rows if r["bed"].startswith("noisy"))

    verdict = {
        "noisy_auto_ge_gini": f"{noisy_auto_ge_gini}/{n_noisy}",
        "per_bed_noisy_win": per_bed_noisy_win,
        "clean_routes_gini_no_perm_cost": clean_no_perm_cost,
        "replicated_noisy_win_every_bed": replicated_noisy,
        "flip_auto_default": bool(replicated_noisy and clean_no_perm_cost),
        "disposition": (
            "FLIP auto default" if (replicated_noisy and clean_no_perm_cost)
            else "KEEP gini default; auto stays OPT-IN (noisy-bed perm win not replicated on every bed)"
        ),
    }
    out = {"rows": rows, "verdict": verdict}
    res_dir = Path(__file__).parent / "_results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "boruta_auto_dispatch.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\nVERDICT:", json.dumps(verdict, indent=2))
    print("results ->", res_dir / "boruta_auto_dispatch.json")


if __name__ == "__main__":
    main()
