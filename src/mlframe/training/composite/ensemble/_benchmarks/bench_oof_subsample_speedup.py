"""Bench: does a GROUP-AWARE subsample of the honest-OOF training rows preserve the NNLS stacking
weights while cutting the K-fold refit wall?

Motivation: on a prod TVT run the cross-target ensemble's honest-OOF stacking refit K folds x ~12
components on ~2.96M rows and took ~4.5 HOURS. The NNLS / dummy-floor weights it produces are a
~12-dim convex-ish solve -- statistically saturated far below millions of rows. This bench measures
(a) the OOF wall scaling with n_train, and (b) whether weights + ensemble RMSE from a group-aware
subsample (whole groups kept) match the full-data weights closely enough to ship the subsample as
the default.

Conclusion feeds the ``oof_max_train_rows`` cap: pick the smallest cap whose weight cosine stays
>=~0.999 and ensemble-RMSE delta stays within noise.

Usage::

    python -m mlframe.training.composite.ensemble._benchmarks.bench_oof_subsample_speedup
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

from mlframe.training.composite import compute_oof_holdout_predictions


def _grouped_dataset(n: int, n_groups: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    levels = rng.uniform(-3.0, 3.0, n_groups)
    groups = rng.integers(0, n_groups, size=n)
    f0 = levels[groups] + rng.normal(0, 0.5, n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    y = 1.5 * f0 + 0.7 * f1 - 0.4 * f2 + rng.normal(0, 0.3, n)
    return X, y.astype(np.float64), groups.astype(np.int64)


def _group_aware_subsample(idx: np.ndarray, groups: np.ndarray, cap: int, seed: int):
    """Keep WHOLE groups until ~cap rows are collected (mirrors how a group-aware split must subsample)."""
    if idx.size <= cap:
        return idx
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups[idx]))
    keep_groups, total = [], 0
    for g in uniq:
        gsz = int(np.sum(groups[idx] == g))
        keep_groups.append(g)
        total += gsz
        if total >= cap:
            break
    mask = np.isin(groups[idx], keep_groups)
    return idx[mask]


def _nnls_weights(oof_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(oof_matrix).all(axis=1) & np.isfinite(y)
    w, _ = nnls(oof_matrix[finite], y[finite])
    s = w.sum()
    return w / s if s > 0 else w


def _oof_call(X, y, groups, kfold):
    models = [Ridge(alpha=a).fit(X, y) for a in (0.1, 1.0, 10.0, 100.0)]
    names = [f"c{i}" for i in range(len(models))]
    specs = [None] * len(models)
    t0 = time.perf_counter()
    oof, _holdout, _surv = compute_oof_holdout_predictions(
        component_models=models, component_names=names, component_specs=specs,
        train_X=X, y_train_full=y, base_train_full_per_spec={},
        holdout_frac=0.2, random_state=42, kfold=kfold, group_ids=groups,
    )
    wall = time.perf_counter() - t0
    return oof, wall


def run(ns=(50_000, 100_000, 200_000), cap=30_000, n_groups=40, kfold=5) -> list[dict]:
    rows = []
    for n in ns:
        X, y, groups = _grouped_dataset(n, n_groups)
        idx = np.arange(n)

        oof_full, wall_full = _oof_call(X, y, groups, kfold)
        w_full = _nnls_weights(oof_full, y)

        sub = _group_aware_subsample(idx, groups, cap, seed=1)
        Xs = X.iloc[sub].reset_index(drop=True)
        oof_sub, wall_sub = _oof_call(Xs, y[sub], groups[sub], kfold)
        w_sub = _nnls_weights(oof_sub, y[sub])

        cos = float(np.dot(w_full, w_sub) / (np.linalg.norm(w_full) * np.linalg.norm(w_sub) + 1e-12))
        # Ensemble RMSE on the FULL OOF surface using each weight vector (apples-to-apples target).
        finite = np.isfinite(oof_full).all(axis=1) & np.isfinite(y)
        rmse_full = float(np.sqrt(np.mean((oof_full[finite] @ w_full - y[finite]) ** 2)))
        rmse_sub = float(np.sqrt(np.mean((oof_full[finite] @ w_sub - y[finite]) ** 2)))
        rows.append({
            "n": n, "cap": cap, "n_sub": int(sub.size), "kfold": kfold,
            "wall_full_s": round(wall_full, 3), "wall_sub_s": round(wall_sub, 3),
            "speedup": round(wall_full / wall_sub, 2) if wall_sub > 0 else None,
            "weight_cosine": round(cos, 6),
            "rmse_full_weights": round(rmse_full, 5), "rmse_sub_weights": round(rmse_sub, 5),
            "rmse_delta_pct": round(100.0 * (rmse_sub - rmse_full) / rmse_full, 4),
            "w_full": [round(float(x), 4) for x in w_full],
            "w_sub": [round(float(x), 4) for x in w_sub],
        })
        print(json.dumps(rows[-1]))
    return rows


if __name__ == "__main__":
    out = run()
    print("\n=== summary", datetime.now().isoformat(timespec="seconds"), "===")
    for r in out:
        print(f"n={r['n']:>7} sub={r['n_sub']:>6}: {r['speedup']}x faster, " f"weight_cos={r['weight_cosine']}, rmse_delta={r['rmse_delta_pct']}%")
