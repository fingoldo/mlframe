"""Empirical grounding study: does the FI-weighted drift score predict MLP harm?

User asked 2026-05-22 whether the ``feature_drift_report`` sensor's
``weighted_drift_score = sum(|z_i| * |fi_i|) / sum(|fi_i|)`` actually
correlates with MLP catastrophic extrapolation, or whether it's a
speculative signal.

Hypothesis (H0): WEIGHTED drift (per-feature z-score scaled by feature
importance) predicts ``MLP_excess_harm`` -- the gap between Ridge_test_R^2
and MLP_test_R^2 on out-of-distribution test data. Ridge handles drift
via linear extrapolation; MLP collapses beyond ~3 std OOD.

Experimental design
-------------------
- Synthetic regression: y = sum_i (alpha_i * x_i) + small Gaussian noise.
- K=5 features. ONE is dominant (alpha=10); 4 are noise (alpha=0.1).
- Per-trial: pick one feature to drift, vary drift_z in
  {0, 1, 2, 3, 5, 7, 10, 15, 20}. Train on shift=0, test on shift=z*train_std.
- 30 trials per (drift_target, drift_z) cell across 10 seeds.
- Models: sklearn Ridge + sklearn MLPRegressor (CPU, fast).
- Per trial measure:
    * weighted_drift_score using GROUND-TRUTH alpha as FI proxy.
    * R^2_test for Ridge and MLP.
    * MLP_excess_harm = Ridge_R^2_test - MLP_R^2_test.

Outputs
-------
- ``profiling/_results/bench_drift_fi_vs_model_harm_<stamp>.csv`` -- per-trial.
- Summary stdout: Pearson r, threshold precision/recall table, verdict.

Run::

    python -m mlframe.profiling.bench_drift_fi_vs_model_harm
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Make repo src importable when run as a plain script.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


N_FEATURES: int = 5
ALPHAS_DOMINANT = np.array([10.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
N_TRAIN: int = 2000
N_TEST: int = 500
NOISE_STD: float = 1.0
DRIFT_Z_LEVELS: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0)
DRIFT_TARGETS = ("none", "dominant", "noise")  # drift NONE / dominant / non-dominant feature
N_SEEDS: int = 10
N_TRIALS_PER_CELL: int = 3  # so 10 seeds * 3 = 30 effective per (target, z) cell

# Two MLP variants: small + medium. Document which one reproduces the prod-log
# pathology better.
MLP_HIDDEN = (32, 16)
MLP_MAX_ITER = 200


def _build_trial(rng: np.random.Generator, drift_target: str, drift_z: float):
    """Build one (X_train, y_train, X_test, y_test) trial with a deterministic
    drift on the requested feature. y = alpha . x + noise; drift shifts the
    test-only x_i mean by drift_z*train_std."""
    alphas = ALPHAS_DOMINANT.copy()

    # Train data ~ N(0, 1) per feature.
    X_train = rng.normal(0.0, 1.0, (N_TRAIN, N_FEATURES))
    y_train = X_train @ alphas + rng.normal(0.0, NOISE_STD, N_TRAIN)

    # Test data ~ N(0, 1) per feature, but ONE feature shifted by drift_z stds.
    X_test = rng.normal(0.0, 1.0, (N_TEST, N_FEATURES))
    if drift_target == "dominant":
        X_test[:, 0] += drift_z
    elif drift_target == "noise":
        X_test[:, 1] += drift_z
    # else "none" -- no shift
    y_test = X_test @ alphas + rng.normal(0.0, NOISE_STD, N_TEST)

    return X_train, y_train, X_test, y_test, alphas


def _weighted_drift_score(X_train: np.ndarray, X_test: np.ndarray, alphas: np.ndarray) -> float:
    """Replicate the production sensor's ``weighted_drift_score`` using
    ground-truth alphas as FI."""
    train_means = X_train.mean(axis=0)
    train_stds = X_train.std(axis=0)
    test_means = X_test.mean(axis=0)
    z = np.where(train_stds > 0, (test_means - train_means) / train_stds, 0.0)
    fi = np.abs(alphas)
    num = float(np.sum(np.abs(z) * fi))
    den = float(np.sum(fi))
    return num / den if den > 0 else 0.0


def _run_one_trial(
    rng: np.random.Generator, drift_target: str, drift_z: float,
) -> dict:
    X_train, y_train, X_test, y_test, alphas = _build_trial(rng, drift_target, drift_z)

    # Pre-pipeline (StandardScaler, same as prod stack).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Ridge baseline.
    ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
    ridge_pred_test = ridge.predict(X_test_s)
    ridge_r2 = float(r2_score(y_test, ridge_pred_test))

    # MLP. Match the prod MLP shape (small declining net + LeakyReLU substitute via relu).
    mlp = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN,
        max_iter=MLP_MAX_ITER,
        random_state=int(rng.integers(0, 1_000_000)),
        early_stopping=True,
        n_iter_no_change=20,
    ).fit(X_train_s, y_train)
    mlp_pred_test = mlp.predict(X_test_s)
    mlp_r2 = float(r2_score(y_test, mlp_pred_test))

    weighted_drift = _weighted_drift_score(X_train, X_test, alphas)

    return {
        "drift_target": drift_target,
        "drift_z": float(drift_z),
        "weighted_drift_score": weighted_drift,
        "ridge_test_r2": ridge_r2,
        "mlp_test_r2": mlp_r2,
        "mlp_excess_harm": ridge_r2 - mlp_r2,
        "ridge_pred_std": float(np.std(ridge_pred_test)),
        "mlp_pred_std": float(np.std(mlp_pred_test)),
        "y_test_std": float(np.std(y_test)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    sx, sy = np.std(x), np.std(y)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    print()
    print("# bench_drift_fi_vs_model_harm  (does FI-weighted drift predict MLP harm?)")
    print(f"# N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, K={N_FEATURES}, alphas={ALPHAS_DOMINANT.tolist()}")
    print(f"# drift_targets={DRIFT_TARGETS}, drift_z_levels={DRIFT_Z_LEVELS}")
    print(f"# seeds=range({N_SEEDS}), trials_per_seed={N_TRIALS_PER_CELL}")
    print()

    rows: list[dict] = []
    t_start = time.perf_counter()
    for seed in range(N_SEEDS):
        for target in DRIFT_TARGETS:
            for z in DRIFT_Z_LEVELS:
                if target == "none" and z != 0.0:
                    continue  # "none" doesn't have a drift level
                for trial in range(N_TRIALS_PER_CELL):
                    rng = np.random.default_rng(seed * 1000 + trial)
                    row = _run_one_trial(rng, target, z)
                    row.update({"seed": seed, "trial": trial})
                    rows.append(row)
    elapsed = time.perf_counter() - t_start
    print(f"# {len(rows)} trials in {elapsed:.1f}s")
    print()

    # Cell summary.
    print(f"{'target':>10} {'drift_z':>7} {'w_drift_mean':>13} {'ridge_R2_mean':>14} {'mlp_R2_mean':>12} {'mlp_excess_mean':>16}")
    print("-" * 80)
    cell_keys = sorted({(r["drift_target"], r["drift_z"]) for r in rows})
    for tgt, z in cell_keys:
        slc = [r for r in rows if r["drift_target"] == tgt and r["drift_z"] == z]
        wd = np.mean([r["weighted_drift_score"] for r in slc])
        r2_ridge = np.mean([r["ridge_test_r2"] for r in slc])
        r2_mlp = np.mean([r["mlp_test_r2"] for r in slc])
        excess = np.mean([r["mlp_excess_harm"] for r in slc])
        print(f"{tgt:>10} {z:>7.1f} {wd:>13.3f} {r2_ridge:>14.3f} {r2_mlp:>12.3f} {excess:>16.3f}")
    print()

    # Overall correlation.
    wd_arr = np.array([r["weighted_drift_score"] for r in rows], dtype=np.float64)
    excess_arr = np.array([r["mlp_excess_harm"] for r in rows], dtype=np.float64)
    r_overall = _pearson(wd_arr, excess_arr)

    # Pearson per drift_target (separates the "drift on dominant" vs "drift on noise" regimes).
    for tgt in DRIFT_TARGETS:
        slc = [r for r in rows if r["drift_target"] == tgt]
        if len(slc) < 3:
            continue
        wd_t = np.array([r["weighted_drift_score"] for r in slc], dtype=np.float64)
        ex_t = np.array([r["mlp_excess_harm"] for r in slc], dtype=np.float64)
        r_t = _pearson(wd_t, ex_t)
        print(f"# Pearson(weighted_drift, MLP_excess_harm | drift_target={tgt}): r = {r_t:+.3f}  (n={len(slc)})")
    print(f"# Pearson(weighted_drift, MLP_excess_harm) overall: r = {r_overall:+.3f}  (n={len(rows)})")
    print()

    # Threshold analysis: at weighted_drift >= 1.0 (the sensor's WARN escalation
    # threshold), what fraction of trials have meaningful MLP_excess_harm (>0.1)?
    def _confusion(threshold_wd: float, threshold_harm: float):
        pred_pos = wd_arr >= threshold_wd
        actual_pos = excess_arr >= threshold_harm
        tp = int(np.sum(pred_pos & actual_pos))
        fp = int(np.sum(pred_pos & ~actual_pos))
        fn = int(np.sum(~pred_pos & actual_pos))
        tn = int(np.sum(~pred_pos & ~actual_pos))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        return tp, fp, fn, tn, precision, recall

    print(f"# Threshold analysis (MLP_excess_harm > 0.1 = 'meaningful harm')")
    print(f"{'w_drift_thr':>12} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'precision':>10} {'recall':>8}")
    print("-" * 60)
    for thr in (0.5, 1.0, 2.0, 3.0, 5.0):
        tp, fp, fn, tn, p, r = _confusion(thr, 0.1)
        print(f"{thr:>12.1f} {tp:>5d} {fp:>5d} {fn:>5d} {tn:>5d} {p:>10.3f} {r:>8.3f}")
    print()

    # Save CSV.
    out_dir = _HERE / "_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_drift_fi_vs_model_harm_{stamp}.csv"
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {out_path}")

    # Verdict.
    print()
    print("# VERDICT")
    abs_r = abs(r_overall) if np.isfinite(r_overall) else 0.0
    if abs_r >= 0.7:
        verdict = "STRONG correlation: weighted_drift_score IS a grounded signal. Auto-action layer can be considered."
    elif abs_r >= 0.4:
        verdict = "MODERATE correlation: signal is partially grounded; defer auto-action; keep WARN escalation."
    else:
        verdict = "WEAK correlation: signal NOT grounded as a standalone predictor; rely on K=2 catastrophic-dropout (target-aware)."
    print(f"#   overall |r| = {abs_r:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
