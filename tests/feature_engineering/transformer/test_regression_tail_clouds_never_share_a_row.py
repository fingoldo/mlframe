"""The top and bottom regression clouds are disjoint even when ties make their quantile bounds coincide."""
import numpy as np

from mlframe.feature_engineering.transformer._utils import class_or_quantile_slice, quantile_tail_masks


def test_tied_target_does_not_put_a_row_in_both_clouds():
    """70% of the rows share one value, so the 20% and 80% quantiles are equal; those rows may sit in one cloud only."""
    y = np.concatenate([np.zeros(70), np.arange(1.0, 31.0)])
    hi, lo = quantile_tail_masks(y, 0.2, 0.8)
    assert not (hi & lo).any()
    X = np.arange(len(y), dtype=np.float32).reshape(-1, 1)
    pos, neg = class_or_quantile_slice(X, y, "regression", 0.8)
    assert not set(pos.ravel()) & set(neg.ravel())
