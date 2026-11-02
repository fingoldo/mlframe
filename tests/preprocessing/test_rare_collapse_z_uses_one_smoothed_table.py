"""The rare-level z-score divides the level-vs-rest log odds ratio by that comparison's own Woolf standard error."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.rare_count_pruning import _target_aware_collapse_mask


def test_large_rare_level_is_judged_against_the_rest_not_the_overall_rate():
    """A level holding 40% of the rows (rate 0.9) against a rest at 0.5: log OR 2.20 / SE 0.186 = 11.8 > 10, so it must be kept.

    Measured against the overall rate (0.66) its offset was only 1.53, z = 8.2 < 10, which collapsed a clearly informative level.
    """
    codes = np.array([0] * 400 + [1] * 600)
    y = np.concatenate([np.r_[np.ones(360), np.zeros(40)], np.r_[np.ones(300), np.zeros(300)]])
    uniques = np.array(["rare", "common"])
    mask = _target_aware_collapse_mask(codes, uniques, y, pd.Index(["rare"]), alpha=0.5, z_threshold=10.0)
    assert not mask[0]
