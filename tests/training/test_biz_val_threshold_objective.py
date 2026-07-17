"""biz_value: decision-threshold tuning objective default = "balanced_accuracy".

The default ``tune_decision_threshold_metric`` was flipped "f1" -> "balanced_accuracy"
because, tuned on a val/OOF split and applied to a fresh test split, balanced_accuracy
recovers the Bayes-optimal operating point ~10x closer and wins test balanced-accuracy
in 29/30 imbalance x seed cells (see bench_threshold_objective.py). This test pins the
win so a regression back to "f1" (or a broken balanced_accuracy path) trips the floor.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from mlframe.training.core._setup_helpers import tune_decision_threshold


def _gen(rng, n, pos_rate, sep, hetero):
    """Generates a synthetic binary-classification dataset with tunable separation and heteroscedasticity."""
    norm = pytest.importorskip("scipy.stats").norm
    npos = max(1, round(n * pos_rate))
    nneg = n - npos
    s = np.concatenate([rng.normal(0.0, 1.0, nneg), rng.normal(sep, hetero, npos)])
    y = np.concatenate([np.zeros(nneg, dtype=np.int8), np.ones(npos, dtype=np.int8)])
    d_neg = norm.pdf(s, 0.0, 1.0) * (1 - pos_rate)
    d_pos = norm.pdf(s, sep, hetero) * pos_rate
    p = d_pos / (d_pos + d_neg + 1e-300)
    idx = rng.permutation(n)
    return y[idx], p[idx]


# (pos_rate, sep, hetero) -- imbalanced / heteroscedastic regimes where the objective matters
_SCENARIOS = [(0.20, 1.8, 1.0), (0.05, 2.2, 1.0), (0.20, 1.8, 1.7), (0.05, 2.2, 1.9), (0.02, 2.6, 1.5)]


def test_biz_val_threshold_balanced_accuracy_beats_f1_majority():
    """balanced_accuracy must win test balanced-accuracy in the MAJORITY of imbalance x seed cells.

    Measured 29/30; floor at 20/25 here (5 scenarios x 5 seeds) to absorb seed noise while
    still failing if the objective regresses to f1 or balanced_accuracy is broken.
    """
    wins = 0
    total = 0
    bacc_ba_sum = 0.0
    bacc_f1_sum = 0.0
    for pr, sep, het in _SCENARIOS:
        for seed in range(5):
            rng = np.random.default_rng(7919 * seed + int(pr * 1000) + int(het * 100))
            yv, pv = _gen(rng, 3000, pr, sep, het)
            yt, pt = _gen(rng, 8000, pr, sep, het)
            thr_ba = tune_decision_threshold(yv, pv, metric="balanced_accuracy")
            thr_f1 = tune_decision_threshold(yv, pv, metric="f1")
            ba = balanced_accuracy_score(yt, (pt >= thr_ba).astype(np.int8))
            f1 = balanced_accuracy_score(yt, (pt >= thr_f1).astype(np.int8))
            bacc_ba_sum += ba
            bacc_f1_sum += f1
            wins += ba >= f1 - 1e-9
            total += 1
    assert wins >= 20, f"balanced_accuracy won only {wins}/{total} cells; expected >=20"
    assert bacc_ba_sum / total >= bacc_f1_sum / total + 0.02, (
        f"mean test balanced-acc: ba={bacc_ba_sum / total:.4f} f1={bacc_f1_sum / total:.4f}; expected >=0.02 lift"
    )


def test_biz_val_threshold_default_metric_is_balanced_accuracy():
    """The flipped config default must remain balanced_accuracy (guards a silent revert)."""
    from mlframe.training._model_configs import TrainingBehaviorConfig

    assert TrainingBehaviorConfig().tune_decision_threshold_metric == "balanced_accuracy"
