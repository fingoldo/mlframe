"""Classification mirror of ``bench_drift_fi_vs_model_harm.py``: does the
FI-weighted drift score predict MLP harm on classification targets too?

The regression study showed Pearson(weighted_drift_score, MLP_excess_R^2_harm)
= +0.834 with precision=1.000 / recall=0.883 at threshold=3.0. That
grounded ``WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD = 3.0``. The
classification override family (alpha=1.0 hidden=(32,16) identity) was
later grounded by ``bench_mlp_robustness_sweep_classification.py``, BUT
the threshold itself was inherited from the regression study without
classification-specific validation. This bench closes that gap.

Hypothesis
----------
H0: ``weighted_drift_score`` predicts ``MLP_excess_logloss`` =
``MLP_log_loss - LogReg_log_loss`` (positive = MLP underperforms
LogReg). If the regression-grounded threshold of 3.0 carries to
classification with comparable precision/recall, the existing
threshold is safe to keep. If precision/recall degrade significantly,
either the threshold needs adjustment or a separate classification
threshold is warranted.

Design
------
- 3 DGPs from the classification sweep (linear_binary,
  interaction_binary, sinusoidal_binary). Multiclass excluded because
  the AUC math is binary-only and we want metric symmetry across DGPs.
- 9 drift_z levels in the sigmoid-transition + saturation range
  {0, 1, 2, 3, 5, 7, 10, 15, 20}. Saturated regimes (z >= 10) are
  included so the harm signal at saturation can be measured.
- N_SEEDS=10 with N_TRIALS_PER_CELL=3 -> 30 trials per cell, 810 trials
  total (matches the regression paired-study sizing).
- weighted_drift_score uses GROUND-TRUTH alphas as FI proxy (same as
  regression bench).
- Models: sklearn ``LogisticRegression`` + ``MLPClassifier`` with
  default ``activation='relu', hidden_layer_sizes=(100,), alpha=1e-4``
  -- the baseline that the override is designed to RESCUE. This bench
  measures the harm BEFORE the override; the override-rescue is
  measured by the companion classification HPT sweep.

Outputs
-------
- ``profiling/_results/bench_drift_fi_vs_model_harm_classification_<stamp>.csv``
- Stdout: Pearson r, threshold precision/recall table, verdict.

Run::

    python -m mlframe.profiling.bench_drift_fi_vs_model_harm_classification
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


N_FEATURES = 5
# alpha_dom=1.0 keeps sigmoid in transition for moderate drift; saturation
# kicks in at z >= 10. Matches the sweep bench design.
ALPHAS_DOMINANT = np.array([1.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
N_TRAIN = 2000
N_TEST = 500
NOISE_STD = 1.0
LOGISTIC_TEMP = 3.0
DRIFT_Z_LEVELS: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0)
DRIFT_TARGETS = ("dominant",)  # noise drift is degenerate for classification (no signal)
N_SEEDS = 10
N_TRIALS_PER_CELL = 3
MLP_HIDDEN = (100,)
MLP_MAX_ITER = 200


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _score_linear(rng, X):
    return X @ ALPHAS_DOMINANT + rng.normal(0.0, NOISE_STD, X.shape[0])


def _score_interaction(rng, X):
    return (
        ALPHAS_DOMINANT[0] * X[:, 0]
        + 3.0 * X[:, 0] * X[:, 1]
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


def _score_sinusoidal(rng, X):
    return (
        5.0 * np.sin(X[:, 0])
        + 3.0 * X[:, 0]
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


DGP_SCORES: dict[str, Callable] = {
    "linear_binary": _score_linear,
    "interaction_binary": _score_interaction,
    "sinusoidal_binary": _score_sinusoidal,
}


def _binarise(score_test, score_train, rng):
    """Logistic-noise binarisation referenced to TRAIN-set normalisation."""
    p = _sigmoid(score_test / LOGISTIC_TEMP)
    return (rng.random(p.shape) < p).astype(np.int64)


def _build_trial(rng, dgp_name, drift_z):
    X_train = rng.normal(0.0, 1.0, (N_TRAIN, N_FEATURES))
    X_test = rng.normal(0.0, 1.0, (N_TEST, N_FEATURES))
    X_test[:, 0] += drift_z
    score_fn = DGP_SCORES[dgp_name]
    score_train = score_fn(rng, X_train)
    score_test = score_fn(rng, X_test)
    y_train = _binarise(score_train, score_train, rng)
    y_test = _binarise(score_test, score_train, rng)
    return X_train, y_train, X_test, y_test


def _weighted_drift_score(X_train, X_test, alphas):
    """Replicate ``feature_drift_report.weighted_drift_score`` using ground-
    truth alphas as FI."""
    train_means = X_train.mean(axis=0)
    train_stds = X_train.std(axis=0)
    test_means = X_test.mean(axis=0)
    z = np.where(train_stds > 0, (test_means - train_means) / train_stds, 0.0)
    fi = np.abs(alphas)
    num = float(np.sum(np.abs(z) * fi))
    den = float(np.sum(fi))
    return num / den if den > 0 else 0.0


def _safe_log_loss(y_true, y_proba, n_classes=2):
    labels = list(range(n_classes))
    eps = 1e-15
    p = np.clip(y_proba, eps, 1 - eps)
    return float(log_loss(y_true, p, labels=labels))


def _run_one_trial(rng, dgp_name, drift_z):
    X_train, y_train, X_test, y_test = _build_trial(rng, dgp_name, drift_z)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        logreg = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs").fit(X_train_s, y_train)
        mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            max_iter=MLP_MAX_ITER,
            random_state=int(rng.integers(0, 1_000_000)),
            early_stopping=True, n_iter_no_change=20,
        ).fit(X_train_s, y_train)

    logreg_logloss = _safe_log_loss(y_test, logreg.predict_proba(X_test_s))
    mlp_logloss = _safe_log_loss(y_test, mlp.predict_proba(X_test_s))
    weighted_drift = _weighted_drift_score(X_train, X_test, ALPHAS_DOMINANT)

    return {
        "dgp": dgp_name,
        "drift_z": float(drift_z),
        "weighted_drift_score": weighted_drift,
        "logreg_logloss": logreg_logloss,
        "mlp_logloss": mlp_logloss,
        "mlp_excess_logloss": mlp_logloss - logreg_logloss,
        "n_train_class1": int(y_train.sum()),
        "n_test_class1": int(y_test.sum()),
    }


def _pearson(x, y):
    if x.size < 3:
        return float("nan")
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    print()
    print("# bench_drift_fi_vs_model_harm_classification")
    print(f"#   alphas={ALPHAS_DOMINANT.tolist()} LOGISTIC_TEMP={LOGISTIC_TEMP}")
    print(f"#   DGPs={list(DGP_SCORES.keys())}")
    print(f"#   drift_z_levels={DRIFT_Z_LEVELS}")
    print(f"#   seeds=range({N_SEEDS}), trials_per_seed={N_TRIALS_PER_CELL}")
    print()

    rows: list[dict] = []
    t0 = time.perf_counter()
    for seed in range(N_SEEDS):
        for dgp_name in DGP_SCORES:
            for z in DRIFT_Z_LEVELS:
                for trial in range(N_TRIALS_PER_CELL):
                    rng = np.random.default_rng(seed * 1000 + trial + abs(hash(dgp_name)) % 7919)
                    row = _run_one_trial(rng, dgp_name, z)
                    row["seed"] = seed
                    row["trial"] = trial
                    rows.append(row)
    elapsed = time.perf_counter() - t0
    print(f"# {len(rows)} trials in {elapsed:.1f}s")
    print()

    # Cell summary by (dgp, drift_z).
    print(f"{'dgp':>22} {'drift_z':>7} {'w_drift_mean':>13} {'logreg_ll':>11} {'mlp_ll':>9} {'mlp_excess':>11}")
    print("-" * 80)
    cell_keys = sorted({(r["dgp"], r["drift_z"]) for r in rows})
    for dgp, z in cell_keys:
        slc = [r for r in rows if r["dgp"] == dgp and r["drift_z"] == z]
        wd = np.mean([r["weighted_drift_score"] for r in slc])
        ll_lr = np.mean([r["logreg_logloss"] for r in slc])
        ll_mlp = np.mean([r["mlp_logloss"] for r in slc])
        excess = np.mean([r["mlp_excess_logloss"] for r in slc])
        print(f"{dgp:>22} {z:>7.1f} {wd:>13.3f} {ll_lr:>11.3f} {ll_mlp:>9.3f} {excess:>11.3f}")
    print()

    wd_arr = np.array([r["weighted_drift_score"] for r in rows], dtype=np.float64)
    excess_arr = np.array([r["mlp_excess_logloss"] for r in rows], dtype=np.float64)

    print()
    for dgp in DGP_SCORES:
        slc = [r for r in rows if r["dgp"] == dgp]
        wd_t = np.array([r["weighted_drift_score"] for r in slc])
        ex_t = np.array([r["mlp_excess_logloss"] for r in slc])
        r_t = _pearson(wd_t, ex_t)
        print(f"# Pearson(weighted_drift, MLP_excess_logloss | dgp={dgp}): r = {r_t:+.3f}  (n={len(slc)})")
    r_overall = _pearson(wd_arr, excess_arr)
    print(f"# Pearson(weighted_drift, MLP_excess_logloss) overall: r = {r_overall:+.3f}  (n={len(rows)})")
    print()

    # Threshold analysis: at weighted_drift >= 3.0 (regression-grounded threshold),
    # how reliable is the harm prediction for classification?
    def _confusion(threshold_wd, threshold_harm):
        pred_pos = wd_arr >= threshold_wd
        actual_pos = excess_arr >= threshold_harm
        tp = int(np.sum(pred_pos & actual_pos))
        fp = int(np.sum(pred_pos & ~actual_pos))
        fn = int(np.sum(~pred_pos & actual_pos))
        tn = int(np.sum(~pred_pos & ~actual_pos))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        return tp, fp, fn, tn, precision, recall

    # MLP_excess_logloss > 0.1 = "meaningful harm" (matches regression-bench framing).
    print(f"# Threshold analysis (MLP_excess_logloss > 0.1 = 'meaningful harm')")
    print(f"{'w_drift_thr':>12} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'precision':>10} {'recall':>8}")
    print("-" * 60)
    for thr in (0.5, 1.0, 2.0, 3.0, 5.0):
        tp, fp, fn, tn, p, r = _confusion(thr, 0.1)
        print(f"{thr:>12.1f} {tp:>5d} {fp:>5d} {fn:>5d} {tn:>5d} {p:>10.3f} {r:>8.3f}")
    print()

    out_dir = _HERE / "_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_drift_fi_vs_model_harm_classification_{stamp}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {out_path}")
    print()

    print("# VERDICT")
    abs_r = abs(r_overall) if np.isfinite(r_overall) else 0.0
    if abs_r >= 0.7:
        verdict = (
            "STRONG correlation: weighted_drift_score IS a grounded signal "
            "for classification harm too. The regression-bench threshold of "
            "3.0 transfers to classification."
        )
    elif abs_r >= 0.4:
        verdict = (
            "MODERATE correlation: signal is partially grounded for "
            "classification; the regression threshold of 3.0 may need "
            "adjustment but the FI-weighted score still carries useful info."
        )
    else:
        verdict = (
            "WEAK correlation: the FI-weighted drift score does NOT reliably "
            "predict classification harm. The regression-grounded threshold "
            "does not transfer; consider a separate classification threshold."
        )
    print(f"#   overall |r| = {abs_r:.3f} -> {verdict}")
    print()


if __name__ == "__main__":
    main()
