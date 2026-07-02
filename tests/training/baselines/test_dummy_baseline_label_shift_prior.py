"""Dummy prior / most_frequent baselines under label shift.

The primary ``prior`` / ``most_frequent`` baselines are HONEST no-skill floors:
what a practitioner could predict having seen ONLY the training labels, so they
use the TRAIN prevalence applied unchanged to val/test. Building them from the
eval split's own class marginal is an eval-peek (the baseline "knows" the label
distribution of the very split it is scored on). The eval-distribution reference
is still computed but exposed ONLY under the explicit ``oracle_prior`` name so it
can never be mistaken for the honest baseline. See the rationale in
``_dummy_baseline_classification._compute_classification_baselines``.
"""
from __future__ import annotations

import numpy as np

from mlframe.training.baselines._dummy_baseline_classification import (
    _compute_classification_baselines,
)


class _Cfg:
    random_state = 0
    stratified_n_repeats = 5
    per_group_max_cardinality_ratio = 0.5
    per_group_high_overlap_threshold = 0.9
    per_group_min_val_coverage_pct = 0.0


def test_prior_is_honest_train_floor_under_label_shift():
    # Train is 90% class-0; val/test are 30% class-0 (strong label shift).
    n_tr, n_ev = 1000, 500
    train_y = np.array([0] * 900 + [1] * 100)
    val_y = np.array([0] * 150 + [1] * 350)
    test_y = np.array([0] * 150 + [1] * 350)
    train_X = np.zeros((n_tr, 1))
    val_X = np.zeros((n_ev, 1))
    test_X = np.zeros((n_ev, 1))

    val_probs, test_probs, _ = _compute_classification_baselines(
        target_name="t", train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=train_y, val_y=val_y, test_y=test_y, timestamps_train=None,
        cat_features=None, config=_Cfg(), target_type="binary_classification", n_classes=2,
    )

    # The honest "prior" baseline is the TRAIN prevalence (class-1 prob ~0.10) applied
    # unchanged to val/test -- what a constant predictor fit on train alone would emit.
    val_prior_p1 = float(val_probs["prior"][0, 1])
    assert abs(val_prior_p1 - 0.10) < 1e-9, f"prior must be the honest train floor (0.10), got {val_prior_p1}"
    test_prior_p1 = float(test_probs["prior"][0, 1])
    assert abs(test_prior_p1 - 0.10) < 1e-9, f"prior must be the honest train floor (0.10), got {test_prior_p1}"

    # The eval-split marginal (class-1 prob ~0.70) is the label-informed reference, exposed
    # under the distinct ``oracle_prior`` name so it is never mistaken for the honest floor.
    assert abs(float(val_probs["oracle_prior"][0, 1]) - 0.70) < 1e-9
    assert abs(float(test_probs["oracle_prior"][0, 1]) - 0.70) < 1e-9

    # most_frequent follows the TRAIN majority (class 0), not the eval-split majority.
    assert int(np.argmax(val_probs["most_frequent"][0])) == 0, "most_frequent must follow the honest train majority"
    assert int(np.argmax(test_probs["most_frequent"][0])) == 0


def test_no_label_shift_prior_unchanged():
    """When eval prevalence matches train, the eval-split prior equals the train
    prior (no behaviour change for the no-shift common case)."""
    train_y = np.array([0] * 700 + [1] * 300)
    val_y = np.array([0] * 70 + [1] * 30)  # same 0.30 class-1 prevalence
    val_probs, _, _ = _compute_classification_baselines(
        target_name="t", train_X=np.zeros((1000, 1)), val_X=np.zeros((100, 1)),
        test_X=np.zeros((100, 1)), train_y=train_y, val_y=val_y, test_y=val_y,
        timestamps_train=None, cat_features=None, config=_Cfg(),
        target_type="binary_classification", n_classes=2,
    )
    assert abs(float(val_probs["prior"][0, 1]) - 0.30) < 1e-9
