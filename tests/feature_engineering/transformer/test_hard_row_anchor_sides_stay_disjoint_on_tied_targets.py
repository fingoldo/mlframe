"""The positive and negative anchor sides of the hard-row transformers never share a row when a tied target makes the y quantiles coincide."""
import numpy as np

from mlframe.feature_engineering.transformer.multi_temp_cbhr import _select_hard_anchors


def test_mostly_tied_target_leaves_the_negative_side_empty_instead_of_duplicating_the_positive_one():
    """85% zeros: q20 == q80 == 0. Both sides used to select from the same rows (y >= 0 and y <= 0 overlap on every zero); the low tail is now empty and reported as such."""
    y = np.concatenate([np.zeros(85), np.ones(15)])
    abs_resid = np.random.default_rng(0).random(100)
    _anchors, _pos_empty, neg_empty = _select_hard_anchors(abs_resid, y, "regression", 5)
    assert neg_empty is True
