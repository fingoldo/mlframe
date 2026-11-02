"""AUC helpers normalise non-{0,1} labels the same way everywhere and reject a mismatched weight length."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import fast_roc_auc
from mlframe.metrics._core_auc_brier import bootstrap_auc_distribution_parallel, fast_roc_auc_unstable, fast_roc_curve, make_bootstrap_auc_resampler

_Y = np.array([1, 2, 1, 2, 1, 2, 1, 2])
_S = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])


def test_unstable_auc_matches_fast_roc_auc_for_labels_one_and_two():
    """Labels {1, 2} (positive = 1, the shared convention) give the same AUC from both entry points."""
    assert fast_roc_auc_unstable(_Y, _S) == fast_roc_auc(_Y, _S)


def test_tie_free_bootstrap_resampler_agrees_with_the_full_sample_auc():
    """The identity resample of the tie-free fast path equals the AUC of the data, with labels {1, 2}."""
    resample = make_bootstrap_auc_resampler(_Y, _S)
    assert resample(np.arange(len(_Y))) == fast_roc_auc(_Y, _S)


def test_parallel_bootstrap_distribution_stays_in_unit_interval_for_labels_one_and_two():
    """Positives counted twice used to push the AUC outside [0, 1]."""
    dist = bootstrap_auc_distribution_parallel(_Y, _S, 20, random_state=0)
    assert dist is None or ((dist >= 0).all() and (dist <= 1).all())


def test_roc_curve_rejects_a_short_weight_vector():
    """A weight vector shorter than y_true must raise instead of being read past its end."""
    with pytest.raises(ValueError):
        fast_roc_curve(np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4]), sample_weight=np.array([1.0, 1.0]))
