"""A/B: does carving a report-only slice off the ShapProxied holdout cost selection quality? (audit FS-07)

Arm A: ``report_holdout_fraction=0`` (selection sees the whole holdout; the winner's ``honest_loss`` is a min over
candidates, optimistic). Arm B: ``report_holdout_fraction=0.25`` (selection sees 75% of it; the chosen subset is scored
once on the untouched 25%).

Bed: the biz_val design (5 informative, 4 noise, 2 columns redundant with informative ones), per seed. Per arm and seed:
informative recall, noise kept, a fresh model's Brier on an independent 5000-row test set drawn from the same process
(the quality that matters), and the gap of each reported number to that test loss (the bias FS-07 is about).

Usage: python -m mlframe.feature_selection.shap_proxied_fs._benchmarks.bench_report_holdout_slice SEED [SEED ...]
Appends one JSON line per (seed, arm) to bench_report_holdout_slice.jsonl next to this file.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "bench_report_holdout_slice.jsonl")
INF = {f"inf{i}" for i in range(5)}


def make(seed: int, n: int):
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 5))
    noise = rng.normal(size=(n, 4))
    corr = inf[:, :2] + 0.3 * rng.normal(size=(n, 2))
    X = pd.DataFrame(np.column_stack([inf, noise, corr]), columns=[f"inf{i}" for i in range(5)] + [f"noise{i}" for i in range(4)] + ["corr0", "corr1"])
    logit = 0.9 * inf[:, 0] + 0.8 * inf[:, 1] - 0.7 * inf[:, 2] + 0.6 * inf[:, 3] + 0.4 * inf[:, 4]
    return X, (logit + 0.3 * rng.normal(size=n) > 0).astype(int)


def run(seed: int, fraction: float) -> dict:
    from xgboost import XGBClassifier

    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = make(seed, 3000)
    X_te, y_te = make(10_000 + seed, 5000)
    sel = ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=7, top_n=20, n_splits=3,
                        n_revalidation_models=2, random_state=seed, verbose=False, n_jobs=1, report_holdout_fraction=fraction)
    sel.fit(X, pd.Series(y))
    chosen = list(sel.selected_features_)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed, n_jobs=1).fit(X[chosen], y)
    test_brier = float(np.mean((model.predict_proba(X_te[chosen])[:, 1] - y_te) ** 2))
    rep = sel.shap_proxy_report_
    ranked = (rep.get("revalidation") or {}).get("ranked") or []
    winner = next((d for d in ranked if d.get("honest_loss_selection_optimistic")), ranked[0] if ranked else {})
    return dict(seed=seed, fraction=fraction, recall=len(set(chosen) & INF) / 5, noise_kept=sum(c.startswith("noise") for c in chosen),
                n_selected=len(chosen), test_brier=test_brier, winner_honest_loss=winner.get("honest_loss"),
                report_holdout_loss=(rep.get("report_holdout") or {}).get("loss"))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        for frac in (0.0, 0.25):
            row = run(s, frac)
            with open(OUT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            sys.stdout.write(json.dumps(row) + "\n")
