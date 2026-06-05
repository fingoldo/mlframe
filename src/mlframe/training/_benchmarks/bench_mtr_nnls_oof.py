"""Bench: MTR per-column ensemble weighting -- equal_mean vs val-fit NNLS vs honest-OOF-fit NNLS (A3-14).

The multi-target-regression per-column ensemble (``MTRPerColumnEqualMeanEnsemble``) can learn per-target NNLS
weights. The question is which surface to fit them on:

  * ``equal_mean``: no fit; 1/n_components per (component, target). The current default.
  * ``val_fit``: NNLS on component predictions over the suite's val fold -- but the components were early-stopped
    against val, so this double-dips a biased surface (the same leak as the single-target A3-01).
  * ``oof_fit``: NNLS on honest train-K-fold OOF predictions -- each component re-fit on K-1 folds, predicting the
    held-out fold. No reuse of the early-stopping surface.

Decision rule (project policy): keep ``equal_mean`` default unless honest-OOF NNLS wins (lower test RMSE) on the
MAJORITY of seeds; also report whether val-fit is optimistically biased (looks good in-fold, worse on test).

Synthetic: correlated K-target regression with K=3; components are boosted-tree-like learners that overfit the
fold they are scored on (a DecisionTree at full depth makes the val/oof distinction sharp, mirroring the
early-stopping overfit of real boosters on val).

Usage::

    python -m mlframe.training._benchmarks.bench_mtr_nnls_oof
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


def _make_mtr(seed: int, n: int = 3000, k: int = 3, n_comp: int = 4):
    rng = np.random.default_rng(seed)
    p = 8
    X = rng.normal(size=(n, p))
    W = rng.normal(size=(p, k))
    Y = X @ W + rng.normal(scale=1.0, size=(n, k))
    # Diverse components: each a tree on a different feature subset (so they overfit differently per fold).
    comps = []
    for c in range(n_comp):
        cols = rng.choice(p, size=max(2, p // 2), replace=False)
        comps.append((cols, DecisionTreeRegressor(max_depth=8, random_state=seed + c)))
    return X, Y, comps


def _fit_components(X, Y, comps):
    fitted = []
    for cols, est in comps:
        m = est.__class__(**est.get_params())
        m.fit(X[:, cols], Y)
        fitted.append((cols, m))
    return fitted


def _predict_stack(X, fitted):
    # (n_comp, N, K)
    return np.stack([m.predict(X[:, cols]) for cols, m in fitted], axis=0)


def _per_column_nnls(stack, Y):
    n_comp, N, K = stack.shape
    w = np.zeros((n_comp, K))
    for kk in range(K):
        wk, _ = nnls(stack[:, :, kk].T, Y[:, kk])
        w[:, kk] = wk if wk.sum() > 0 else 1.0 / n_comp
    return w


def _apply(stack, w):
    return np.einsum("cnk,ck->nk", stack, w)


def _rmse(P, Y):
    return float(np.sqrt(np.mean((P - Y) ** 2)))


def _oof_stack(X_train, Y_train, comps, kfold=5, seed=0):
    """Honest train-K-fold OOF prediction stack on the train rows."""
    n, k = Y_train.shape
    n_comp = len(comps)
    oof = np.full((n_comp, n, k), np.nan)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    for tr, ho in kf.split(np.arange(n)):
        fitted = _fit_components(X_train[tr], Y_train[tr], comps)
        for ci, (cols, m) in enumerate(fitted):
            oof[ci, ho, :] = m.predict(X_train[ho][:, cols])
    return oof


def _bench_seed(seed: int) -> dict:
    X, Y, comps = _make_mtr(seed)
    n = X.shape[0]
    # train / val (= ES surface) / test split.
    i1, i2 = int(n * 0.6), int(n * 0.8)
    Xtr, Ytr = X[:i1], Y[:i1]
    Xval, Yval = X[i1:i2], Y[i1:i2]
    Xte, Yte = X[i2:], Y[i2:]

    # Final components fit on train (as the suite does before ensembling).
    fitted = _fit_components(Xtr, Ytr, comps)
    test_stack = _predict_stack(Xte, fitted)

    # equal_mean
    n_comp = len(comps)
    w_eq = np.full((n_comp, Y.shape[1]), 1.0 / n_comp)
    rmse_eq = _rmse(_apply(test_stack, w_eq), Yte)

    # val_fit NNLS (leaky: components saw val for ES analog)
    val_stack = _predict_stack(Xval, fitted)
    w_val = _per_column_nnls(val_stack, Yval)
    rmse_val = _rmse(_apply(test_stack, w_val), Yte)

    # oof_fit NNLS (honest train-K-fold OOF)
    oof = _oof_stack(Xtr, Ytr, comps, kfold=5, seed=seed)
    w_oof = _per_column_nnls(oof, Ytr)
    rmse_oof = _rmse(_apply(test_stack, w_oof), Yte)

    return {
        "seed": seed,
        "rmse_equal_mean": rmse_eq,
        "rmse_val_fit": rmse_val,
        "rmse_oof_fit": rmse_oof,
        "oof_beats_equal": rmse_oof < rmse_eq,
        "oof_beats_val": rmse_oof < rmse_val,
    }


def main() -> None:
    """Benchmark MTR per-column NNLS weights fit on honest K-fold OOF vs equal_mean vs val-fit across seeds; writes the verdict JSON to _results/."""
    seeds = list(range(8))
    rows = [_bench_seed(s) for s in seeds]
    mean_eq = float(np.mean([r["rmse_equal_mean"] for r in rows]))
    mean_val = float(np.mean([r["rmse_val_fit"] for r in rows]))
    mean_oof = float(np.mean([r["rmse_oof_fit"] for r in rows]))
    oof_wins = sum(r["oof_beats_equal"] for r in rows)

    print("MTR per-column ensemble: test RMSE by weighting surface\n")
    print("| seed | equal_mean | val_fit | oof_fit | oof<equal | oof<val |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['seed']} | {r['rmse_equal_mean']:.4f} | {r['rmse_val_fit']:.4f} | "
              f"{r['rmse_oof_fit']:.4f} | {r['oof_beats_equal']} | {r['oof_beats_val']} |")
    print(f"\nmean equal_mean={mean_eq:.4f}  val_fit={mean_val:.4f}  oof_fit={mean_oof:.4f}")
    print(f"oof_fit beats equal_mean on {oof_wins}/{len(seeds)} seeds")
    verdict = "nnls_oof" if oof_wins > len(seeds) / 2 else "equal_mean"
    print(f"DECISION: MTR default = {verdict} (keep equal_mean unless honest-OOF NNLS wins majority).")

    out = {
        "bench": "mtr_nnls_oof",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "mean_equal_mean": mean_eq,
        "mean_val_fit": mean_val,
        "mean_oof_fit": mean_oof,
        "oof_wins": oof_wins,
        "n_seeds": len(seeds),
        "decision_default": verdict,
    }
    _dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(_dir, exist_ok=True)
    _path = os.path.join(_dir, "bench_mtr_nnls_oof.json")
    with open(_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {_path}")


if __name__ == "__main__":
    main()
