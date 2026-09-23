"""The per-group AUC printed beside the pooled AUC must weight groups by their rows.

It was a plain mean over groups, so one 4-row group scoring 1.0 by luck offset a 10,000-row group one for one and the
bracketed figure in the report title read better than the model is.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._auc_per_group import compute_mean_aucs_per_group


def test_a_tiny_lucky_group_does_not_move_the_weighted_mean():
    """A 4-row group scoring a perfect 1.0 must barely move a mean dominated by a 10,000-row group."""
    aucs = {0: (0.60, 0.40), 1: (1.00, 1.00)}
    sizes = {0: 10_000, 1: 4}
    roc, _pr = compute_mean_aucs_per_group(aucs, group_sizes=sizes)
    assert roc == pytest.approx((0.60 * 10_000 + 1.0 * 4) / 10_004)
    assert roc < 0.61


def test_without_sizes_the_mean_stays_unweighted():
    """A caller who passes no sizes keeps the old plain mean rather than silently getting a different statistic."""
    roc, _ = compute_mean_aucs_per_group({0: (0.60, 0.4), 1: (1.0, 1.0)})
    assert roc == pytest.approx(0.80)


def test_nan_groups_are_skipped_in_both_modes():
    """A group whose AUC is undefined contributes neither a value nor its weight, weighted or not."""
    aucs = {0: (0.70, 0.5), 1: (np.nan, np.nan)}
    assert compute_mean_aucs_per_group(aucs, group_sizes={0: 10, 1: 1000})[0] == pytest.approx(0.70)
    assert compute_mean_aucs_per_group(aucs)[0] == pytest.approx(0.70)
