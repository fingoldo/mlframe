"""Multi-seed measurement of the one-SE band against the legacy across-fold-std band.

Same protocol as tests/feature_selection/wrappers/rfecv/test_biz_val_rfecv_one_se_band.py, over more split seeds,
so the 0.85 feature-count bar and the AUC no-harm bar are judged on a distribution rather than two seeds.
Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_rfecv_one_se_seeds
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.wrappers import RFECV
from tests.feature_selection.wrappers.rfecv.test_biz_val_rfecv_one_se_band import _replay

SEEDS = tuple(range(10))


def main() -> None:
    X, y = make_classification(400, 200, n_informative=8, n_redundant=0, class_sep=1.5, random_state=0, shuffle=False)
    X = pd.DataFrame((X - X.mean(0)) / X.std(0), columns=[f"f{i}" for i in range(200)])
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)
            sel = RFECV(estimator=LogisticRegression(C=0.3, max_iter=300), cv=3, random_state=seed, max_refits=8, verbose=0).fit(Xtr, ytr)
            row = {"seed": seed}
            for rule, tag in (("one_se_max_foldstd", "legacy"), ("one_se_max", "se")):
                kept = [c for c, m in zip(X.columns, _replay(sel, rule)) if m]
                row[f"n_{tag}"] = len(kept)
                row[f"auc_{tag}"] = float(roc_auc_score(yte, LogisticRegression(C=0.3, max_iter=300).fit(Xtr[kept], ytr).predict_proba(Xte[kept])[:, 1]))
            rows.append(row)
    n_legacy = np.array([r["n_legacy"] for r in rows], dtype=float)
    n_se = np.array([r["n_se"] for r in rows], dtype=float)
    auc_legacy = np.array([r["auc_legacy"] for r in rows])
    auc_se = np.array([r["auc_se"] for r in rows])
    summary = {
        "ratio_mean_se_over_legacy": float(n_se.mean() / n_legacy.mean()),
        "mean_n_se": float(n_se.mean()),
        "mean_n_legacy": float(n_legacy.mean()),
        "mean_auc_delta_se_minus_legacy": float((auc_se - auc_legacy).mean()),
        "min_auc_delta": float((auc_se - auc_legacy).min()),
        "rows": rows,
    }
    json.dump(summary, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
