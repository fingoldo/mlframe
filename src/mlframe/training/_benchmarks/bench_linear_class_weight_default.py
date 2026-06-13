"""Bench: sklearn linear head class_weight=None vs "balanced" on imbalanced classification.

Question: when the target-distribution analyzer flags class_imbalance (max/min > 10x) it sets
class_weight="balanced" for the lgb/cb/xgb heads but NOT for the sklearn linear heads
(_build_linear/_lasso/_elasticnet/_ridge in models.py), which silently fit unweighted.

This bench measures whether "balanced" wins on the honest holdout for the linear head across
imbalanced synthetics x seeds, on balanced-accuracy and ROC-AUC (the metrics that matter when
one class is rare; plain accuracy is uninformative under heavy imbalance).

Run: python src/mlframe/training/_benchmarks/bench_linear_class_weight_default.py
Pure sklearn / numpy -- no mlframe import (avoids the host cupy-probe segfault).

VERDICT (deferred, NOT flipped): at a FIXED 0.5 threshold, "balanced" wins balanced-accuracy
24/25 (mean +0.1353) but ROC-AUC is a wash (15/25, +0.0025). The flip is NOT shipped because
the bench does not account for the production decision-threshold tuner (tune_decision_threshold,
metric balanced_accuracy): on imbalanced targets the tuner already moves the unweighted model's
threshold toward the minority and recovers most of the balanced-accuracy gain, so the marginal
benefit of class_weight="balanced" ON TOP of tuning is unknown and likely much smaller. AUC
(threshold-free) shows class_weight is ~neutral. A real flip needs a (unweighted+tuned) vs
(balanced+tuned) bench, plus a check that balanced + per-class post-hoc calibration does not
over-correct. The class_weight knob/wiring is held until then.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _scenarios():
    # (name, n, weights[minority frac], n_informative, flip_y, sep)
    return [
        ("imb_20x", 4000, [0.05, 0.95], 6, 0.01, 1.0),
        ("imb_50x", 6000, [0.02, 0.98], 8, 0.01, 1.2),
        ("imb_12x", 3000, [0.08, 0.92], 5, 0.03, 0.8),
        ("imb_100x", 8000, [0.01, 0.99], 8, 0.0, 1.5),
        ("imb_30x_noisy", 5000, [0.033, 0.967], 6, 0.05, 0.7),
    ]


def main():
    seeds = [0, 1, 2, 3, 4]
    wins_bal_acc = 0
    wins_auc = 0
    total = 0
    rows = []
    for name, n, weights, ninfo, flip, sep in _scenarios():
        for seed in seeds:
            X, y = make_classification(
                n_samples=n, n_features=ninfo + 6, n_informative=ninfo,
                n_redundant=2, weights=weights, flip_y=flip, class_sep=sep,
                random_state=seed,
            )
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.33, stratify=y, random_state=seed,
            )
            res = {}
            for cw in (None, "balanced"):
                m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                       class_weight=cw, random_state=seed)
                m.fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                pred = (p >= 0.5).astype(int)
                res[cw] = (
                    balanced_accuracy_score(yte, pred),
                    roc_auc_score(yte, p),
                )
            total += 1
            d_ba = res["balanced"][0] - res[None][0]
            d_auc = res["balanced"][1] - res[None][1]
            if d_ba > 1e-6:
                wins_bal_acc += 1
            if d_auc > 1e-6:
                wins_auc += 1
            rows.append((name, seed, res[None][0], res["balanced"][0], d_ba,
                         res[None][1], res["balanced"][1], d_auc))

    print(f"{'scenario':<16}{'seed':<5}{'ba_none':>9}{'ba_bal':>9}{'dBA':>9}"
          f"{'auc_none':>10}{'auc_bal':>9}{'dAUC':>9}")
    for r in rows:
        print(f"{r[0]:<16}{r[1]:<5}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>+9.4f}"
              f"{r[5]:>10.4f}{r[6]:>9.4f}{r[7]:>+9.4f}")
    mean_dba = np.mean([r[4] for r in rows])
    mean_dauc = np.mean([r[7] for r in rows])
    print(f"\nbalanced wins balanced-accuracy: {wins_bal_acc}/{total}  mean dBA={mean_dba:+.4f}")
    print(f"balanced wins ROC-AUC:          {wins_auc}/{total}  mean dAUC={mean_dauc:+.4f}")


if __name__ == "__main__":
    main()
