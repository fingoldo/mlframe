"""Isolated bench: decision-threshold tuning OBJECTIVE for binary hard labels.

Lever: ``TrainingBehaviorConfig.tune_decision_threshold_metric`` default ("f1" vs
"balanced_accuracy" vs Youden-J). ``tune_decision_threshold`` sweeps a grid and
maximises the chosen objective on a val/OOF split; the recovered threshold is then
applied on a fresh test split. The honest question: which objective, tuned on val,
yields the best test-split BALANCED ACCURACY (the imbalance-robust hard-label metric
the tuning exists to recover) across imbalance regimes -- and how close does each get
to the population Bayes-optimal balanced-accuracy threshold?

Run:
    python src/mlframe/training/_benchmarks/bench_threshold_objective.py

HONEST metric: test-split balanced accuracy (model never tuned on it) + |thr - thr*|
where thr* is the population balanced-accuracy-optimal threshold on a large oracle pool.
Decision flips the default ONLY if a challenger wins the MAJORITY of scenario x seed
cells on test balanced accuracy. Otherwise KEEP + record the reject verdict here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

from mlframe.training.core._setup_helpers import tune_decision_threshold  # noqa: E402

N_CAND = 200


def _youden_threshold(y: np.ndarray, p: np.ndarray, default: float = 0.5) -> float:
    """Challenger objective: maximise Youden's J = TPR - FPR over the same grid."""
    if y.shape[0] == 0 or np.unique(y).shape[0] < 2 or not np.all(np.isfinite(p)):
        return default
    cand = np.linspace(0.0, 1.0, N_CAND + 2)[1:-1]
    pos = y == 1
    neg = ~pos
    npos = pos.sum()
    nneg = neg.sum()
    best_j = -np.inf
    best = default
    for thr in cand:
        pred = p >= thr
        tpr = (pred & pos).sum() / npos
        fpr = (pred & neg).sum() / nneg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best = float(thr)
    return best


def _gen(rng: np.random.Generator, n: int, pos_rate: float, sep: float, hetero: float):
    """Heteroscedastic 1D-score synthetic with a known calibrated P(y=1).

    Latent score s; positives shifted by ``sep``; class-conditional std scaled by
    ``hetero`` for the positive class (heteroscedastic). ``p`` is the TRUE posterior
    from the two Gaussians (so 0.5 on p is Bayes-optimal for accuracy, but NOT for
    balanced accuracy under imbalance -- that is the whole point of tuning).
    """
    npos = max(1, int(round(n * pos_rate)))
    nneg = n - npos
    s_neg = rng.normal(0.0, 1.0, nneg)
    s_pos = rng.normal(sep, hetero, npos)
    s = np.concatenate([s_neg, s_pos])
    y = np.concatenate([np.zeros(nneg, dtype=np.int8), np.ones(npos, dtype=np.int8)])
    # true posterior via Gaussian densities + prior
    from scipy.stats import norm

    d_neg = norm.pdf(s, 0.0, 1.0) * (1 - pos_rate)
    d_pos = norm.pdf(s, sep, hetero) * pos_rate
    p = d_pos / (d_pos + d_neg + 1e-300)
    idx = rng.permutation(n)
    return y[idx], p[idx]


def _oracle_thr(rng, pos_rate, sep, hetero):
    """Population balanced-accuracy-optimal threshold from a huge pool."""
    y, p = _gen(rng, 400_000, pos_rate, sep, hetero)
    return _youden_threshold(y, p)  # Youden-J == balanced-acc optimum (analytically)


SCENARIOS = [
    # (name, pos_rate, sep, hetero)
    ("balanced_homo", 0.50, 1.6, 1.0),
    ("mild_imb_homo", 0.20, 1.8, 1.0),
    ("rare_homo", 0.05, 2.2, 1.0),
    ("mild_imb_hetero", 0.20, 1.8, 1.7),
    ("rare_hetero", 0.05, 2.2, 1.9),
    ("extreme_rare", 0.02, 2.6, 1.5),
]
SEEDS = [0, 1, 2, 3, 4]
N_VAL = 3000
N_TEST = 8000


def run():
    objectives = ["f1", "balanced_accuracy", "youden"]
    cells = {o: [] for o in objectives}
    rows = []
    for name, pr, sep, het in SCENARIOS:
        thr_star = _oracle_thr(np.random.default_rng(99), pr, sep, het)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * seed + hash(name) % 997)
            yv, pv = _gen(rng, N_VAL, pr, sep, het)
            yt, pt = _gen(rng, N_TEST, pr, sep, het)
            thrs = {
                "f1": tune_decision_threshold(yv, pv, metric="f1", n_candidates=N_CAND),
                "balanced_accuracy": tune_decision_threshold(yv, pv, metric="balanced_accuracy", n_candidates=N_CAND),
                "youden": _youden_threshold(yv, pv),
            }
            for o in objectives:
                bacc = balanced_accuracy_score(yt, (pt >= thrs[o]).astype(np.int8))
                f1v = f1_score(yt, (pt >= thrs[o]).astype(np.int8), zero_division=0)
                cells[o].append(bacc)
                rows.append(
                    {
                        "scenario": name,
                        "seed": seed,
                        "objective": o,
                        "thr": round(thrs[o], 4),
                        "thr_star": round(float(thr_star), 4),
                        "abs_thr_err": round(abs(thrs[o] - thr_star), 4),
                        "test_bacc": round(float(bacc), 4),
                        "test_f1": round(float(f1v), 4),
                    }
                )

    # majority vote per cell on test balanced accuracy
    n_cells = len(SCENARIOS) * len(SEEDS)
    wins = {o: 0 for o in objectives}
    ties = 0
    for i in range(n_cells):
        scores = {o: cells[o][i] for o in objectives}
        mx = max(scores.values())
        winners = [o for o in objectives if scores[o] >= mx - 1e-9]
        if len(winners) == len(objectives):
            ties += 1
        for w in winners:
            wins[w] += 1

    summary = {
        "n_cells": n_cells,
        "mean_test_bacc": {o: round(float(np.mean(cells[o])), 4) for o in objectives},
        "mean_abs_thr_err": {o: round(float(np.mean([r["abs_thr_err"] for r in rows if r["objective"] == o])), 4) for o in objectives},
        "cell_wins_test_bacc": wins,
        "all_tie_cells": ties,
        "current_default": "f1",
    }
    out_dir = Path(__file__).resolve().parent / "_results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "threshold_objective.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
