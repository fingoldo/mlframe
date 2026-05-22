"""Classification-DGP MLP HPT sweep -- extends the regression-only override
to classification targets.

Companion to ``bench_mlp_robustness_sweep_nonlinear.py``. The earlier
sweep stack used 100% regression DGPs; the wire-in in
``_phase_train_one_target_body`` therefore gates the override on
``_is_regression_target_type``. This bench closes that gap by sweeping
the same Cartesian HPT space on classification DGPs and identifying the
cross-DGP min-max winner under log-loss, accuracy, and (for binary)
ROC AUC.

DGPs
----
- ``linear_binary``: y = (linear_score > median) where linear_score has
  one dominant feature (alphas=[10, 0.1, 0.1, 0.1, 0.1]).
- ``interaction_binary``: y = (10*x_dom + 3*x_dom*x_2 > median).
- ``sinusoidal_binary``: y = (5*sin(x_dom) + 3*x_dom > median).
- ``linear_multiclass_3``: 3-class via score quantile cuts on the same
  linear score.

Each DGP shifts the dominant feature by drift_z stds at test time, same
as the regression sweep.

Baseline reference
------------------
Regression used Ridge as the "linear, drift-resilient" baseline. For
classification the natural baseline is sklearn LogisticRegression. We
compute ``MLP_log_loss_excess = LogReg_log_loss - MLP_log_loss`` (note
sign: higher MLP_log_loss is worse, so positive excess = MLP underperforms
LogReg). Same for accuracy / ROC AUC.

Metrics
-------
- ``log_loss`` (lower = better; primary classification metric)
- ``accuracy`` (higher = better)
- ``roc_auc`` (binary only, higher = better)

Output
------
- ``profiling/_results/bench_mlp_robustness_sweep_classification_<stamp>.csv``
- Per-DGP leaderboards under each metric
- Cross-DGP min-max winner under each metric

Run::

    python -m mlframe.profiling.bench_mlp_robustness_sweep_classification
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


N_FEATURES: int = 5
# Scaled down to keep the sigmoid in its transition region under drift.
# Regression's [10, 0.1, ...] would shift the linear score by alpha_dom *
# drift_z stds; at alpha=10, drift_z=10 the score moves by 100, fully
# saturating sigmoid(score/TEMP) and degenerating one class to 0 in test.
# alpha_dom=1.0 + drift_z=5 keeps sigmoid argument ~1.67, in transition.
ALPHAS_DOMINANT = np.array([1.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
N_TRAIN: int = 2000
N_TEST: int = 500
NOISE_STD: float = 1.0
MLP_MAX_ITER: int = 200
# Logistic-noise temperature: y ~ Bernoulli(sigmoid(score / TEMP)). Higher
# = noisier labels = more room for both models to disagree under drift.
LOGISTIC_TEMP: float = 3.0

ALPHA_GRID: tuple[float, ...] = (1e-4, 1e-1, 1.0, 10.0)
HIDDEN_GRID: tuple[tuple[int, ...], ...] = ((100,), (32, 16), (16,), (8,))
ACTIVATION_GRID: tuple[str, ...] = ("relu", "tanh", "identity")

PHASE1_DRIFT_Z: float = 5.0  # classification sigmoid-transition regime
PHASE1_DRIFT_TARGET: str = "dominant"
PHASE1_N_SEEDS: int = 15

PHASE2_DRIFT_Z_LEVELS: Sequence[float] = (0.0, 2.0, 5.0, 8.0)
PHASE2_DRIFT_TARGETS = ("dominant",)
PHASE2_N_SEEDS: int = 8
PHASE2_TOP_K: int = 4


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _to_class_logistic(score: np.ndarray, train_scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Logistic-noise binarisation. y ~ Bernoulli(sigmoid(score / TEMP)).
    Unlike the brittle median-threshold approach this keeps both classes
    populated under heavy drift -- the sigmoid saturates instead of
    flipping all labels at once."""
    p = _sigmoid(score / LOGISTIC_TEMP)
    return (rng.random(p.shape) < p).astype(np.int64)


def _to_class_3way_softmax(score: np.ndarray, train_scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """3-class via softmax over three score-shifted "logits". Class
    boundaries shift smoothly with drift instead of all-collapsing to one
    class once the score crosses a fixed threshold."""
    train_mean = float(np.mean(train_scores))
    train_std = float(np.std(train_scores)) or 1.0
    logits = np.column_stack([
        -(score - train_mean) / train_std,                    # class 0: low scores
        np.full_like(score, 0.0),                             # class 1: middle scores
        (score - train_mean) / train_std,                     # class 2: high scores
    ])
    # Stable softmax along axis=1.
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    # Sample 3-way categorical per row.
    u = rng.random(probs.shape[0])
    cum = np.cumsum(probs, axis=1)
    out = np.zeros(probs.shape[0], dtype=np.int64)
    out[u > cum[:, 0]] = 1
    out[u > cum[:, 1]] = 2
    return out


def _score_linear(rng, X):
    return X @ ALPHAS_DOMINANT + rng.normal(0.0, NOISE_STD, X.shape[0])


def _score_interaction(rng, X):
    return (
        10.0 * X[:, 0]
        + 3.0 * X[:, 0] * X[:, 1]
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


def _score_sinusoidal(rng, X):
    return (
        5.0 * np.sin(X[:, 0])
        + 3.0 * X[:, 0]
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


DGPs: dict[str, dict] = {
    "linear_binary": {
        "score": _score_linear, "label": _to_class_logistic, "n_classes": 2,
    },
    "interaction_binary": {
        "score": _score_interaction, "label": _to_class_logistic, "n_classes": 2,
    },
    "sinusoidal_binary": {
        "score": _score_sinusoidal, "label": _to_class_logistic, "n_classes": 2,
    },
    "linear_multiclass_3": {
        "score": _score_linear, "label": _to_class_3way_softmax, "n_classes": 3,
    },
}


def _build_trial(
    rng: np.random.Generator, drift_target: str, drift_z: float, dgp_name: str,
):
    spec = DGPs[dgp_name]
    X_train = rng.normal(0.0, 1.0, (N_TRAIN, N_FEATURES))
    X_test = rng.normal(0.0, 1.0, (N_TEST, N_FEATURES))
    if drift_target == "dominant":
        X_test[:, 0] += drift_z
    elif drift_target == "noise":
        X_test[:, 1] += drift_z
    score_train = spec["score"](rng, X_train)
    score_test = spec["score"](rng, X_test)
    # Logistic / softmax-noise labellers take rng for Bernoulli/categorical
    # sampling. Production binning is fixed at train time so the label
    # function references train_scores for normalisation.
    y_train = spec["label"](score_train, score_train, rng)
    y_test = spec["label"](score_test, score_train, rng)
    return X_train, y_train, X_test, y_test, spec["n_classes"]


def _safe_log_loss(y_true, y_proba, n_classes):
    """log_loss with class-label hint so a test set missing a class
    doesn't raise; clip probs to avoid -inf."""
    labels = list(range(n_classes))
    eps = 1e-15
    p = np.clip(y_proba, eps, 1 - eps)
    return float(log_loss(y_true, p, labels=labels))


def _safe_roc_auc(y_true, y_proba, n_classes):
    """ROC AUC for binary only. Returns NaN otherwise or if y_true is
    single-class (which can happen after drift wipes out one class)."""
    if n_classes != 2:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    return float(roc_auc_score(y_true, proba_pos))


def _run_one_trial(
    rng: np.random.Generator, drift_target: str, drift_z: float, dgp_name: str,
    alpha: float, hidden: tuple[int, ...], activation: str,
) -> dict:
    X_train, y_train, X_test, y_test, n_classes = _build_trial(
        rng, drift_target, drift_z, dgp_name,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # multinomial-friendly logistic regression with light L2; lbfgs
        # solver auto-handles binary vs multiclass without the deprecated
        # multi_class kwarg.
        logreg = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs",
        ).fit(X_train_s, y_train)
    logreg_proba = logreg.predict_proba(X_test_s)
    logreg_pred = logreg.predict(X_test_s)
    logreg_logloss = _safe_log_loss(y_test, logreg_proba, n_classes)
    logreg_acc = float(accuracy_score(y_test, logreg_pred))
    logreg_auc = _safe_roc_auc(y_test, logreg_proba, n_classes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=activation,
            alpha=alpha,
            max_iter=MLP_MAX_ITER,
            random_state=int(rng.integers(0, 1_000_000)),
            early_stopping=True,
            n_iter_no_change=20,
        ).fit(X_train_s, y_train)
    mlp_proba = mlp.predict_proba(X_test_s)
    mlp_pred = mlp.predict(X_test_s)
    mlp_logloss = _safe_log_loss(y_test, mlp_proba, n_classes)
    mlp_acc = float(accuracy_score(y_test, mlp_pred))
    mlp_auc = _safe_roc_auc(y_test, mlp_proba, n_classes)

    # excess_harm positive = MLP underperforms LogReg under this metric.
    # For log_loss / acc the sign flips (acc: lower MLP is worse, so
    # excess = logreg_acc - mlp_acc).
    return {
        "dgp": dgp_name,
        "drift_target": drift_target,
        "drift_z": float(drift_z),
        "alpha": alpha,
        "hidden": str(hidden),
        "activation": activation,
        "n_classes": int(n_classes),
        "logreg_logloss": logreg_logloss,
        "mlp_logloss": mlp_logloss,
        "logreg_acc": logreg_acc,
        "mlp_acc": mlp_acc,
        "logreg_auc": logreg_auc,
        "mlp_auc": mlp_auc,
        "mlp_excess_logloss": mlp_logloss - logreg_logloss,
        "mlp_excess_acc": logreg_acc - mlp_acc,
        "mlp_excess_auc": (logreg_auc - mlp_auc) if np.isfinite(logreg_auc) and np.isfinite(mlp_auc) else float("nan"),
    }


METRICS = ("mlp_excess_logloss", "mlp_excess_acc", "mlp_excess_auc")
METRIC_LABELS = {
    "mlp_excess_logloss": "log_loss gap (MLP - LogReg)",
    "mlp_excess_acc": "accuracy gap (LogReg - MLP)",
    "mlp_excess_auc": "ROC-AUC gap (binary; LogReg - MLP)",
}


def _config_key(alpha: float, hidden: tuple[int, ...], activation: str) -> str:
    return f"alpha={alpha:g} hidden={hidden} activation={activation}"


def _phase1_per_dgp(dgp_name: str):
    rows: list[dict] = []
    by_cfg: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    for i, (alpha, hidden, activation) in enumerate(
            product(ALPHA_GRID, HIDDEN_GRID, ACTIVATION_GRID), start=1):
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in range(PHASE1_N_SEEDS):
            rng = np.random.default_rng(seed * 31 + i + abs(hash(dgp_name)) % 7919)
            row = _run_one_trial(
                rng, PHASE1_DRIFT_TARGET, PHASE1_DRIFT_Z, dgp_name,
                alpha, hidden, activation,
            )
            row["seed"] = seed
            row["phase"] = "1"
            rows.append(row)
            for m in METRICS:
                v = row[m]
                if np.isfinite(v):
                    per_metric[m].append(v)
        cfg = {"alpha": alpha, "hidden_layer_sizes": hidden, "activation": activation}
        by_cfg[(alpha, hidden, activation)] = {
            "cfg": cfg,
            "key": _config_key(alpha, hidden, activation),
            "means": {
                m: float(np.mean(vs)) if vs else float("nan")
                for m, vs in per_metric.items()
            },
        }
    elapsed = time.perf_counter() - t0
    print(f"# Phase 1 [{dgp_name}] done in {elapsed:.1f}s "
          f"({len(ALPHA_GRID) * len(HIDDEN_GRID) * len(ACTIVATION_GRID)} configs x "
          f"{PHASE1_N_SEEDS} seeds = {len(rows)} trials)")
    leaderboards: dict[str, list[tuple[str, float, dict]]] = {}
    for m in METRICS:
        ranked = [(e["key"], e["means"][m], e["cfg"]) for e in by_cfg.values()]
        # NaN sorts last (cross-DGP NaN cells == AUC for multiclass).
        ranked.sort(key=lambda r: (np.isnan(r[1]), r[1]))
        leaderboards[m] = ranked
    return rows, leaderboards


def _phase2_per_dgp(dgp_name: str, winners: dict[str, list[tuple[str, float, dict]]]):
    baseline_cfg = {
        "alpha": ALPHA_GRID[0], "hidden_layer_sizes": HIDDEN_GRID[0],
        "activation": ACTIVATION_GRID[0],
    }
    baseline_key = _config_key(
        baseline_cfg["alpha"], baseline_cfg["hidden_layer_sizes"], baseline_cfg["activation"],
    )
    seen: set[str] = set()
    keep: list[tuple[str, float, dict]] = []
    for metric_board in winners.values():
        for entry in metric_board[:PHASE2_TOP_K]:
            if entry[0] not in seen:
                seen.add(entry[0])
                keep.append(entry)
    if baseline_key not in seen:
        keep.append((baseline_key, float("nan"), baseline_cfg))

    rows: list[dict] = []
    t0 = time.perf_counter()
    for ci, (_k, _h, cfg) in enumerate(keep, start=1):
        for tgt in PHASE2_DRIFT_TARGETS:
            for z in PHASE2_DRIFT_Z_LEVELS:
                for seed in range(PHASE2_N_SEEDS):
                    rng = np.random.default_rng(seed * 97 + ci * 7 + abs(hash(dgp_name)) % 7919)
                    row = _run_one_trial(
                        rng, tgt, z, dgp_name,
                        cfg["alpha"], cfg["hidden_layer_sizes"], cfg["activation"],
                    )
                    row["seed"] = seed
                    row["phase"] = "2"
                    rows.append(row)
    elapsed = time.perf_counter() - t0
    print(f"# Phase 2 [{dgp_name}] done in {elapsed:.1f}s ({len(rows)} trials)")
    return rows


def _print_leaderboard(dgp_name, leaderboards, top=8):
    for m, board in leaderboards.items():
        print()
        print(f"# Phase 1 [{dgp_name}] -- ranked by {METRIC_LABELS[m]} (lower=better; top {top})")
        print(f"{'rank':>4} {'config':<55} {'mean':>14}")
        print("-" * 80)
        for i, (k, val, _c) in enumerate(board[:top], start=1):
            print(f"{i:>4} {k:<55} {val:>14.4f}")
    print()


def _pick_cross_dgp_winner_for_metric(metric, per_dgp_leaderboards):
    boards = {
        dgp: lbs[metric] for dgp, lbs in per_dgp_leaderboards.items() if metric in lbs
    }
    if not boards:
        return None, None
    all_keys = set(k for lb in boards.values() for k, _h, _c in lb)
    rows = []
    for key in all_keys:
        per_dgp: dict[str, float] = {}
        cfg = None
        for dgp_name, lb in boards.items():
            for k, h, c in lb:
                if k == key:
                    per_dgp[dgp_name] = h
                    cfg = c
                    break
        # Filter out DGPs where this metric is NaN (e.g. AUC on multiclass)
        # before computing the cross-DGP min-max so 3-class DGPs don't
        # disqualify every config under the AUC metric.
        finite = {n: v for n, v in per_dgp.items() if np.isfinite(v)}
        if len(finite) >= 2:  # need at least 2 DGPs for min-max to mean something
            worst = max(finite.values())
            mean = float(np.mean(list(finite.values())))
            rows.append((key, worst, mean, per_dgp, cfg))
    rows.sort(key=lambda r: (r[1], r[2]))

    print()
    print(f"# Cross-DGP min-max -- {METRIC_LABELS[metric]} (top 10 by worst-case across applicable DGPs)")
    print(f"{'rank':>4} {'config':<55} {'worst':>10} {'avg':>10} per_dgp")
    print("-" * 100)
    for i, (k, worst, mean, per_dgp, _c) in enumerate(rows[:10], start=1):
        per_dgp_str = " ".join(
            f"{n[:7]}={(v if np.isfinite(v) else float('nan')):.3f}" for n, v in per_dgp.items()
        )
        print(f"{i:>4} {k:<55} {worst:>10.4f} {mean:>10.4f} {per_dgp_str}")
    print()
    if not rows:
        return None, None
    return rows[0][0], rows[0][4]


def main():
    print()
    print("# bench_mlp_robustness_sweep_classification")
    print(f"#   DGPs={list(DGPs.keys())}")
    print(f"#   alpha_grid={ALPHA_GRID}")
    print(f"#   hidden_grid={HIDDEN_GRID}")
    print(f"#   activation_grid={ACTIVATION_GRID}")
    print(f"#   phase1: drift_z={PHASE1_DRIFT_Z} on '{PHASE1_DRIFT_TARGET}', "
          f"{PHASE1_N_SEEDS} seeds per cfg")

    all_rows: list[dict] = []
    per_dgp_leaderboards: dict[str, dict[str, list[tuple[str, float, dict]]]] = {}
    for dgp_name in DGPs:
        p1_rows, lbs = _phase1_per_dgp(dgp_name)
        all_rows.extend(p1_rows)
        per_dgp_leaderboards[dgp_name] = lbs
        _print_leaderboard(dgp_name, lbs)

    for dgp_name, lbs in per_dgp_leaderboards.items():
        p2_rows = _phase2_per_dgp(dgp_name, lbs)
        all_rows.extend(p2_rows)

    out_dir = _HERE / "_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_mlp_robustness_sweep_classification_{stamp}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"# wrote {out_path}")
    print()

    per_metric_winners: dict[str, tuple] = {}
    for metric in METRICS:
        per_metric_winners[metric] = _pick_cross_dgp_winner_for_metric(
            metric, per_dgp_leaderboards,
        )

    print("# CROSS-METRIC SUMMARY -- classification (cross-DGP min-max winners)")
    print(f"{'metric':<32} {'winner':<60}")
    print("-" * 95)
    for metric, (k, _c) in per_metric_winners.items():
        print(f"{METRIC_LABELS[metric]:<32} {k or '(none)':<60}")
    print()

    keys = {k for k, _c in per_metric_winners.values() if k}
    if len(keys) == 1:
        print(f"# AGREEMENT: all metrics agree on winner -> {keys.pop()}")
    else:
        print(f"# DIVERGENCE: metrics disagree, candidates={keys}")
    print()


if __name__ == "__main__":
    main()
