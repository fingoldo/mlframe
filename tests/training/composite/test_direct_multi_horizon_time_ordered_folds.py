"""With time_ordered=True the block-size search purges training rows next to each test block (tracker 2026-11-02 wave 15, LEAK15-5)."""

import numpy as np
from sklearn.linear_model import Ridge

from mlframe.training.composite.direct_multi_horizon import DirectMultiHorizonEnsemble


def test_time_ordered_folds_keep_a_horizon_gap_between_train_and_test():
    """Y[t, h] = Y[t+1, h-1] for consecutive origins, so train rows within n_horizons - 1 of a test block would share its targets."""
    est = DirectMultiHorizonEnsemble(lambda: Ridge(), auto_block_search=True, cv_splits=3, time_ordered=True)
    for train_idx, test_idx in est._block_search_folds(60, 6):
        gap = np.min(np.abs(train_idx[:, None] - test_idx[None, :]), axis=1)
        assert gap.min() >= 6


def test_default_folds_are_still_shuffled():
    """time_ordered=False keeps the shuffled K-fold: test indices of the first fold are not one contiguous block."""
    est = DirectMultiHorizonEnsemble(lambda: Ridge(), auto_block_search=True, cv_splits=3, random_state=0)
    first_test = next(iter(est._block_search_folds(60, 6)))[1]
    assert np.diff(np.sort(first_test)).max() > 1
