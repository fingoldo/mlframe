"""SA30 regression: dummy prior / most_frequent baselines under label shift.

Pre-fix the ``prior`` and ``most_frequent`` no-skill baselines used the TRAIN
prevalence scored on the val/test split. Under label shift (train prevalence !=
eval prevalence) that gives a biased no-skill floor. The fix computes the
prevalence on the EVALUATION split, so the floor reflects the distribution it is
scored against.
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


def test_prior_reflects_eval_split_under_label_shift():
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

    # prior baseline row is constant per split; it must equal the EVAL prevalence
    # (class-1 prob ~0.70), NOT the train prevalence (class-1 prob 0.10).
    val_prior_p1 = float(val_probs["prior"][0, 1])
    assert abs(val_prior_p1 - 0.70) < 1e-9, f"val prior must reflect eval split (0.70), got {val_prior_p1}"
    test_prior_p1 = float(test_probs["prior"][0, 1])
    assert abs(test_prior_p1 - 0.70) < 1e-9, f"test prior must reflect eval split (0.70), got {test_prior_p1}"

    # most_frequent on the eval split is class 1 (majority on eval), not class 0
    # (the train majority). Pre-fix it predicted class 0 (train argmax).
    assert int(np.argmax(val_probs["most_frequent"][0])) == 1, "most_frequent must follow eval-split majority"
    assert int(np.argmax(test_probs["most_frequent"][0])) == 1


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
