"""Unit + biz_value tests for ordinal cutpoint optimization (PZAD minfunc)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._ordinal_cutpoints import (
    apply_cutpoints,
    optimal_ordinal_cutpoints,
)
from mlframe.metrics.classification._weighted_kappa import quadratic_weighted_kappa


# ---------------------------------------------------------------- unit
def test_apply_cutpoints_basic():
    yp = np.array([0.1, 1.4, 2.4, 3.9])  # land inside bins [<.5], [.5,1.5), [1.5,2.5), [2.5,inf)
    labels = apply_cutpoints(yp, np.array([0.5, 1.5, 2.5]), n_classes=4)
    assert labels.tolist() == [0, 1, 2, 3]


def test_apply_cutpoints_clips_to_range():
    yp = np.array([-5.0, 100.0])
    labels = apply_cutpoints(yp, np.array([0.5, 1.5]), n_classes=3)
    assert labels.tolist() == [0, 2]


def test_perfectly_separable_recovers_labels():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0.1, 0.2, 1.1, 1.2, 2.1, 2.2])  # already separable
    thr, score = optimal_ordinal_cutpoints(y, pred, n_classes=3, metric="qwk")
    assert score > 0.99
    assert np.array_equal(apply_cutpoints(pred, thr, 3), y)


def test_invalid_metric_and_nclasses_and_mismatch():
    with pytest.raises(ValueError):
        optimal_ordinal_cutpoints(np.zeros(3, int), np.zeros(3), n_classes=3, metric="nope")
    with pytest.raises(ValueError):
        optimal_ordinal_cutpoints(np.zeros(3, int), np.zeros(3), n_classes=1)
    with pytest.raises(ValueError):
        optimal_ordinal_cutpoints(np.zeros(3, int), np.zeros(2), n_classes=3)


# ---------------------------------------------------------------- biz_value
def test_biz_val_tuned_cutpoints_beat_naive_rounding_on_qwk():
    """The CrowdFlower technique: a regression fit on an ordinal grade + tuned cutpoints beats naive round() at QWK.
    Naive rounding assumes the grades sit at integer centers; a shifted/scaled regression output does not."""
    rng = np.random.default_rng(0)
    n = 2000
    n_classes = 5
    y = rng.integers(0, n_classes, size=n)
    # regression output correlated with grade but shifted + scaled + noisy (as a real model would be)
    pred = 0.6 * y + 0.8 + rng.normal(0, 0.5, size=n)
    naive = np.clip(np.round(pred), 0, n_classes - 1).astype(int)
    qwk_naive = quadratic_weighted_kappa(y, naive, n_classes=n_classes)
    _thr, qwk_tuned = optimal_ordinal_cutpoints(y, pred, n_classes=n_classes, metric="qwk")
    assert qwk_tuned >= qwk_naive + 0.02, f"tuned QWK {qwk_tuned:.3f} should beat naive-round {qwk_naive:.3f}"


def test_biz_val_optimizer_never_worse_than_quantile_warmstart():
    """The returned thresholds are guaranteed >= the prevalence-quantile warm start (the guard in the optimizer)."""
    rng = np.random.default_rng(1)
    n = 1500
    n_classes = 4
    y = rng.integers(0, n_classes, size=n)
    pred = y + rng.normal(0, 0.9, size=n)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    cum = np.cumsum(counts)[:-1] / n
    warm = np.quantile(pred, cum)
    qwk_warm = quadratic_weighted_kappa(y, apply_cutpoints(pred, warm, n_classes), n_classes=n_classes)
    _, qwk_opt = optimal_ordinal_cutpoints(y, pred, n_classes=n_classes, metric="qwk")
    assert qwk_opt >= qwk_warm - 1e-9
