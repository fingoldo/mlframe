"""ICE subsamples the learn set (only printed) but always scores the eval set early stopping reads in full."""

from __future__ import annotations

import numpy as np

from mlframe.metrics._ice_metric import ICE
from mlframe.metrics.calibration import ICE_UNCOMPUTABLE


def _metric(y_true, y_score):
    """Constant metric returning 0.25."""
    return 0.25


def _call(m, n, seed=0):
    """Evaluate the metric on n random scores and labels and return the first result."""
    rng = np.random.default_rng(seed)
    return m.evaluate([rng.normal(size=n)], (rng.random(n) < 0.3).astype(np.int8), None)[0]


def test_largest_set_is_skipped_only_after_a_second_size_appears():
    """Largest set is skipped only after a second size appears."""
    m = ICE(metric=_metric, higher_is_better=False, skip_largest_set=True, subsample_skipped_sets=False)
    assert _call(m, 1000) == 0.25, "the only set seen so far must still be scored"
    assert _call(m, 200) == 0.25, "the smaller (eval) set is always scored"
    assert _call(m, 1000) == ICE_UNCOMPUTABLE, "with the legacy opt-out, the larger (learn) set is skipped once an eval set is known"
    assert _call(m, 200) == 0.25


def test_largest_set_is_subsampled_by_default():
    """Largest set is subsampled by default."""
    sizes = []

    def _rec(y_true, y_score):
        """Record the size of the set being scored and return a constant."""
        sizes.append(len(y_true))
        return 0.25

    m = ICE(metric=_rec, higher_is_better=False, skip_largest_set=True, learn_sample_size=300)
    assert _call(m, 1000) == 0.25 and _call(m, 200) == 0.25 and _call(m, 1000) == 0.25 and _call(m, 200) == 0.25
    assert sizes == [1000, 200, 300, 200], "the learn set is scored on a subsample once an eval set is known; the eval set never is"


def test_single_set_run_keeps_its_metric():
    """Single set run keeps its metric."""
    m = ICE(metric=_metric, higher_is_better=False, skip_largest_set=True)
    assert len(range(5)) > 0, "the loop below must iterate at least once"
    for _ in range(5):
        assert _call(m, 5000) == 0.25


def test_flag_off_scores_everything():
    """Flag off scores everything."""
    m = ICE(metric=_metric, higher_is_better=False)
    assert _call(m, 5000) == 0.25 and _call(m, 100) == 0.25 and _call(m, 5000) == 0.25
