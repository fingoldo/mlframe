"""BoundSplitter lets a group-aware splitter drive the OOF loop, which only ever calls ``splitter.split(X)``."""
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from mlframe.feature_engineering.transformer._utils import BoundSplitter


def test_bare_group_kfold_raises_but_the_bound_one_keeps_groups_whole():
    """``GroupKFold().split(X)`` needs groups; bound, no group straddles the train/validation boundary."""
    X = np.zeros((60, 2))
    groups = np.repeat(np.arange(12), 5)
    with pytest.raises(ValueError):
        list(GroupKFold(n_splits=3).split(X))
    bound = BoundSplitter(GroupKFold(n_splits=3), groups=groups)
    assert bound.n_splits == 3
    folds = list(bound.split(X))
    assert len(folds) == 3
    for tr, va in folds:
        assert not set(groups[tr]) & set(groups[va])
