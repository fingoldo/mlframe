"""ICE skips the learn set (only printed) but never the eval set early stopping reads."""

from __future__ import annotations

import numpy as np

from mlframe.metrics._ice_metric import ICE


def _metric(y_true, y_score):
    return 0.25


def _call(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return m.evaluate([rng.normal(size=n)], (rng.random(n) < 0.3).astype(np.int8), None)[0]


def test_largest_set_is_skipped_only_after_a_second_size_appears():
    m = ICE(metric=_metric, higher_is_better=False, skip_largest_set=True)
    assert _call(m, 1000) == 0.25, "the only set seen so far must still be scored"
    assert _call(m, 200) == 0.25, "the smaller (eval) set is always scored"
    assert _call(m, 1000) == 0, "the larger (learn) set is skipped once an eval set is known"
    assert _call(m, 200) == 0.25


def test_single_set_run_keeps_its_metric():
    m = ICE(metric=_metric, higher_is_better=False, skip_largest_set=True)
    for _ in range(5):
        assert _call(m, 5000) == 0.25


def test_flag_off_scores_everything():
    m = ICE(metric=_metric, higher_is_better=False)
    assert _call(m, 5000) == 0.25 and _call(m, 100) == 0.25 and _call(m, 5000) == 0.25
