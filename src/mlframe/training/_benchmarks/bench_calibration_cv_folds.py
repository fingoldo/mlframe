"""Multi-scenario bench: CalibratedClassifierCV inner-fold count default.

Lever: ``DEFAULT_CALIBRATION_CV_FOLDS`` in ``training/models.py`` (currently 3),
used by ``_make_calibrated_classifier_cv`` with ``method="isotonic"``.

Isotonic calibration is data-hungry: with k folds each calibrator fits the
base model on (k-1)/k of TRAIN and learns the isotonic map on the held-out
1/k. More folds -> each base model sees more data (less under-fit) and the
ensemble averages more calibrators, but each isotonic map is learned on a
smaller slice (noisier step function). The honest question is which k gives
the lowest calibration error / log-loss on a TRUE holdout the calibrator
never saw.

HONEST metric: Brier score + log-loss on a held-out TEST split (never used
for fit or calibration). Lower is better. We also report ROC-AUC (rank
quality, should be ~invariant to calibration) as a sanity guard.

Decision rule (CLAUDE.md variant-defaults / accuracy-first): flip the default
only if an alternative wins on the MAJORITY of (scenario x seed) cells on the
honest Brier metric. Single-seed wins do not count.

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.bench_calibration_cv_folds
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

FOLD_CANDIDATES = (3, 5)  # current default 3 vs alternative 5
SEEDS = (0, 1, 2, 3)


def _make_scenario(name: str, seed: int):
    """Return (X, y, base_estimator) for a scenario. Base models chosen
    poorly-calibrated by construction so calibration has work to do."""
    rng = seed
    if name == "balanced_mid":
        X, y = make_classification(n_samples=2400, n_features=20, n_informative=8, n_redundant=4, weights=[0.5, 0.5], class_sep=0.8, random_state=rng)
        base = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=rng)
    elif name == "imbalanced_15pct":
        X, y = make_classification(n_samples=2800, n_features=25, n_informative=10, n_redundant=5, weights=[0.85, 0.15], class_sep=0.9, random_state=rng)
        base = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=rng)
    elif name == "small_n":
        X, y = make_classification(n_samples=900, n_features=15, n_informative=6, n_redundant=3, weights=[0.6, 0.4], class_sep=0.7, random_state=rng)
        base = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=rng)
    elif name == "high_dim_noisy":
        X, y = make_classification(n_samples=2000, n_features=60, n_informative=12, n_redundant=10, weights=[0.5, 0.5], class_sep=0.5, random_state=rng)
        base = GradientBoostingClassifier(n_estimators=70, max_depth=4, random_state=rng)
    elif name == "logreg_overconfident":
        X, y = make_classification(n_samples=2200, n_features=30, n_informative=10, n_redundant=8, weights=[0.7, 0.3], class_sep=0.6, random_state=rng)
        base = LogisticRegression(C=50.0, max_iter=400, random_state=rng)
    else:
        raise ValueError(name)
    return X, y, base


def _eval_fold_count(X, y, base, k, seed):
    X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    clf = CalibratedClassifierCV(clone(base), cv=inner_cv, method="isotonic")
    clf.fit(X_fit, y_fit)
    p = clf.predict_proba(X_test)[:, 1]
    return {
        "brier": float(brier_score_loss(y_test, p)),
        "logloss": float(log_loss(y_test, np.clip(p, 1e-7, 1 - 1e-7))),
        "auc": float(roc_auc_score(y_test, p)),
    }


def main():
    scenarios = ["balanced_mid", "imbalanced_15pct", "small_n", "high_dim_noisy", "logreg_overconfident"]
    rows = []
    wins = {k: 0 for k in FOLD_CANDIDATES}
    cells = 0
    for sc in scenarios:
        for seed in SEEDS:
            X, y, base = _make_scenario(sc, seed)
            res = {k: _eval_fold_count(X, y, base, k, seed) for k in FOLD_CANDIDATES}
            # honest winner = lowest Brier
            best_k = min(FOLD_CANDIDATES, key=lambda k: res[k]["brier"])
            wins[best_k] += 1
            cells += 1
            rows.append({"scenario": sc, "seed": seed,
                         **{f"brier_k{k}": round(res[k]["brier"], 5) for k in FOLD_CANDIDATES},
                         **{f"logloss_k{k}": round(res[k]["logloss"], 5) for k in FOLD_CANDIDATES},
                         **{f"auc_k{k}": round(res[k]["auc"], 5) for k in FOLD_CANDIDATES},
                         "brier_winner": best_k})
            print(f"{sc:22s} seed={seed}  "
                  + "  ".join(f"k{k} brier={res[k]['brier']:.5f}" for k in FOLD_CANDIDATES)
                  + f"  -> k{best_k}")

    mean_brier = {k: float(np.mean([r[f"brier_k{k}"] for r in rows])) for k in FOLD_CANDIDATES}
    mean_logloss = {k: float(np.mean([r[f"logloss_k{k}"] for r in rows])) for k in FOLD_CANDIDATES}
    cur, alt = 3, 5
    majority_alt = wins[alt] > cells / 2
    verdict = ("FLIP to k=%d" % alt) if majority_alt else "KEEP k=3"
    summary = {
        "lever": "DEFAULT_CALIBRATION_CV_FOLDS (CalibratedClassifierCV isotonic inner folds)",
        "candidates": list(FOLD_CANDIDATES),
        "cells": cells,
        "brier_wins": wins,
        "mean_brier": {str(k): round(v, 5) for k, v in mean_brier.items()},
        "mean_logloss": {str(k): round(v, 5) for k, v in mean_logloss.items()},
        "majority_rule_winner_k5": majority_alt,
        "verdict": verdict,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "calibration_cv_folds.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
