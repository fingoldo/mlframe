"""Regression: the primary classification dummy prior/most_frequent baselines are an
HONEST no-skill floor fit on the TRAIN marginal only (never the eval split's own class
marginal). The eval-distribution reference is surfaced separately as ``oracle_prior``.

Guards against the eval-peek foot-gun where ``prior`` / ``most_frequent`` were built from
``_prior_from(val_y)`` / ``_prior_from(test_y)`` -- a label-informed, optimistic floor the
model itself never saw.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.baselines._dummy_baseline_classification import (
    _compute_classification_baselines,
)


def _cfg():
    return SimpleNamespace(
        random_state=0,
        stratified_n_repeats=3,
        per_group_max_cardinality_ratio=0.5,
        per_group_high_overlap_threshold=0.9,
        per_group_min_val_coverage_pct=50.0,
    )


def test_prior_uses_train_marginal_not_eval_marginal():
    # Deliberate label shift: train is 80/20, val is 25/75, test is 50/50.
    train_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    val_y = np.array([0, 1, 1, 1])
    test_y = np.array([1, 1, 0, 0])
    train_prior = np.array([0.8, 0.2])

    val_probs, test_probs, _ = _compute_classification_baselines(
        "t", None, None, None, train_y, val_y, test_y, None, None, _cfg(),
        "binary_classification", 2,
    )

    # HONEST prior == train marginal on BOTH eval splits (not the split's own marginal).
    np.testing.assert_allclose(val_probs["prior"][0], train_prior)
    np.testing.assert_allclose(test_probs["prior"][0], train_prior)
    # It must NOT equal the eval-split marginals (which would be the eval-peek bug).
    assert not np.allclose(val_probs["prior"][0], [0.25, 0.75])
    assert not np.allclose(test_probs["prior"][0], [0.5, 0.5])

    # most_frequent argmax follows the TRAIN majority class (0), regardless of eval majority.
    assert int(np.argmax(val_probs["most_frequent"][0])) == 0
    assert int(np.argmax(test_probs["most_frequent"][0])) == 0


def test_oracle_prior_exposes_eval_marginal_under_distinct_name():
    train_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    val_y = np.array([0, 1, 1, 1])
    test_y = np.array([1, 1, 0, 0])

    val_probs, test_probs, _ = _compute_classification_baselines(
        "t", None, None, None, train_y, val_y, test_y, None, None, _cfg(),
        "binary_classification", 2,
    )

    # The label-informed reference lives ONLY under oracle_prior, and DOES reflect the
    # eval split's own marginal (so a reviewer can see it, but never confuse it with prior).
    np.testing.assert_allclose(val_probs["oracle_prior"][0], [0.25, 0.75])
    np.testing.assert_allclose(test_probs["oracle_prior"][0], [0.5, 0.5])
    # oracle_prior differs from the honest prior under label shift.
    assert not np.allclose(val_probs["oracle_prior"][0], val_probs["prior"][0])
