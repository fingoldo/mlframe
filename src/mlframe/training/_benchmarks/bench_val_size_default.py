"""Honest-holdout multi-scenario bench for the TrainingSplitConfig.val_size default.

val_size is the early-stop / eval-set carve fraction taken from the train portion. The
current default is 0.1. Too small a val makes the early-stop signal noisy; too large
wastes training rows. This bench sweeps val_size in {0.1, 0.15, 0.2} over 6 scenarios
(clf + reg, varied n and feature count) x 3 seeds and scores on an EXTERNAL honest
holdout never seen by the suite. A variant flips the default only if it wins the
majority of (scenario, seed) cells vs the 0.1 baseline.

Each suite run gets a UNIQUE model_name per (scenario, mode, seed, variant) because the
workdir is shared -- colliding dirs across different feature-count scenarios made predict
load a wrong-feature-space model (16 != 20 LightGBM error). Do not regress that.

Run (host env vars required to avoid the cupy-probe segfault on this box):
  set CUDA_VISIBLE_DEVICES= & set MLFRAME_NO_CUDA_AUTOCONFIG=1 & set MLFRAME_KEEP_BROKEN_CUPY=1
  python -m mlframe.training._benchmarks.bench_val_size_default
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from mlframe.training.core import train_mlframe_models_suite, predict_mlframe_models_suite
from mlframe.training import TrainingSplitConfig, OutputConfig

try:
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor
except Exception:  # pragma: no cover - allow running from installed tree
    from mlframe.tests.training.shared import SimpleFeaturesAndTargetsExtractor

VAL_SIZES = [0.10, 0.15, 0.20]

# (name, mode, n_train, n_test, n_features). Moderate n where the val-carve fraction
# actually changes the early-stop budget; clf + reg both represented.
SCENARIOS = [
    ("reg_small", "reg", 1500, 500, 16),
    ("reg_mid", "reg", 3000, 800, 24),
    ("clf_small", "clf", 1500, 500, 16),
    ("clf_mid", "clf", 3000, 800, 24),
    ("clf_noisy", "clf", 2500, 700, 20),
    ("reg_wide", "reg", 2500, 700, 40),
]
SEEDS = [42, 7, 99]


def _make_regression(n_train, n_test, n_features, seed):
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.randn(n, n_features)
    k = min(6, n_features)
    coefs = np.zeros(n_features)
    coefs[:k] = rng.uniform(-3, 3, size=k)
    y = X @ coefs + rng.randn(n) * 1.5
    cols = [f"f_{i}" for i in range(n_features)]
    tr = pd.DataFrame(X[:n_train], columns=cols); tr["target"] = y[:n_train]
    te = pd.DataFrame(X[n_train:], columns=cols); te["target"] = y[n_train:]
    return tr, te


def _make_classification(n_train, n_test, n_features, seed, noise=2.0):
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.randn(n, n_features)
    k = min(6, n_features)
    coefs = np.zeros(n_features)
    coefs[:k] = rng.uniform(-1.5, 1.5, size=k)
    logits = X @ coefs + rng.randn(n) * noise
    y = (logits > 0).astype(int)
    cols = [f"f_{i}" for i in range(n_features)]
    tr = pd.DataFrame(X[:n_train], columns=cols); tr["target"] = y[:n_train]
    te = pd.DataFrame(X[n_train:], columns=cols); te["target"] = y[n_train:]
    return tr, te


def _score(scenario, mode, n_tr, n_te, n_feat, seed, val_size, tmp_root):
    regression = mode == "reg"
    if regression:
        train_df, test_df = _make_regression(n_tr, n_te, n_feat, seed)
    else:
        train_df, test_df = _make_classification(n_tr, n_te, n_feat, seed)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression)
    vtag = f"v{int(round(val_size * 100)):03d}"
    model_name = f"{scenario}_{mode}_s{seed}_{vtag}"
    data_dir = str(Path(tmp_root) / model_name)
    train_mlframe_models_suite(
        df=train_df,
        target_name="test_target",
        model_name=model_name,
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        split_config=TrainingSplitConfig(val_size=val_size),
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        hyperparams_config={"iterations": 2000, "early_stopping_rounds": 30},
        verbose=0,
    )
    models_path = f"{data_dir}/models/test_target/{model_name}"
    results = predict_mlframe_models_suite(
        df=test_df,
        models_path=models_path,
        features_and_targets_extractor=fte,
        return_probabilities=not regression,
        verbose=0,
    )
    if regression:
        preds = np.asarray(next(iter(results["predictions"].values())), dtype=float)
        return float(np.sqrt(mean_squared_error(test_df["target"].values, preds)))
    if results.get("probabilities"):
        probs = np.asarray(next(iter(results["probabilities"].values())))
        score_vec = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
    else:
        score_vec = np.asarray(next(iter(results["predictions"].values())), dtype=float)
    return float(roc_auc_score(test_df["target"].values, score_vec))


def main():
    tmp_root = tempfile.mkdtemp(prefix="bench_valsize_")
    cells = {}  # (scenario, seed) -> {val_size: metric}
    for name, mode, n_tr, n_te, n_feat in SCENARIOS:
        for seed in SEEDS:
            row = {}
            for v in VAL_SIZES:
                metric = _score(name, mode, n_tr, n_te, n_feat, seed, v, tmp_root)
                row[v] = metric
                print(f"{name:10s} {mode} seed={seed} val={v:.2f} -> {metric:.5f}")
            cells[f"{name}|{seed}"] = {"mode": mode, **{str(k): v for k, v in row.items()}}

    baseline = 0.10
    summary = {}
    for alt in [x for x in VAL_SIZES if x != baseline]:
        wins = ties = losses = 0
        for key, row in cells.items():
            mode = row["mode"]
            b = row[str(baseline)]
            a = row[str(alt)]
            # reg: lower RMSE is better; clf: higher AUROC is better. 0.2% relative tol = tie.
            if mode == "reg":
                rel = (b - a) / b
            else:
                rel = (a - b) / b
            if rel > 0.002:
                wins += 1
            elif rel < -0.002:
                losses += 1
            else:
                ties += 1
        n = len(cells)
        summary[str(alt)] = {"wins": wins, "ties": ties, "losses": losses, "n": n,
                             "verdict": "FLIP" if wins > n / 2 else "KEEP_DEFAULT"}

    out = {"baseline_val_size": baseline, "cells": cells, "summary": summary}
    res_dir = Path(__file__).parent / "_results"
    res_dir.mkdir(exist_ok=True)
    res_path = res_dir / "val_size_default.json"
    res_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\nSUMMARY (alt vs baseline val_size=0.10):")
    for alt, s in summary.items():
        print(f"  val={alt}: wins={s['wins']} ties={s['ties']} losses={s['losses']} / {s['n']} -> {s['verdict']}")
    print(f"\nwrote {res_path}")
    return out


if __name__ == "__main__":
    main()
