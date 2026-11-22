"""Audit 2026-11-20 finding 25-F37: the per-group AUC fills its doomed groups in one construction and counts their NaNs without a loop.

The values and key order are pinned to the output recorded before the change, on a fixture with a single-sample group and an
all-negative group (both doomed to NaN), so the vectorised fill cannot drift from the old dict.
"""

import math

import numpy as np

from mlframe.metrics._auc_per_group import fast_aucs_per_group_optimized


def test_doomed_groups_are_nan_and_the_rest_match_the_recorded_output():
    """Overall AUCs, the NaN groups 5 and 6 (listed first), and the valid groups 3, 4, 7, 9 with their recorded values."""
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0], dtype=np.float64)
    s = np.array([0.9, 0.1, 0.8, 0.4, 0.3, 0.2, 0.7, 0.6, 0.5, 0.45, 0.05, 0.35, 0.5, 0.2, 0.3])
    g = np.array([7, 7, 7, 3, 3, 9, 9, 9, 4, 4, 4, 4, 5, 6, 6])
    roc, pr, groups = fast_aucs_per_group_optimized(y, s, g)
    assert math.isclose(roc, 0.8928571428571429) and math.isclose(pr, 0.884920634920635)
    assert list(groups.keys()) == [5, 6, 3, 4, 7, 9]
    assert math.isnan(groups[5][0]) and math.isnan(groups[5][1])
    assert math.isnan(groups[6][0]) and math.isnan(groups[6][1])
    assert groups[3] == (1.0, 1.0)
    assert groups[4] == (0.75, 0.8333333333333333)
    assert groups[7] == (1.0, 1.0) and groups[9] == (1.0, 1.0)
