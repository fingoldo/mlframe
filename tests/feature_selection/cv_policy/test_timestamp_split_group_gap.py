"""TimestampOrderedSplit keeps its embargo ``gap`` on the grouped (GroupTimeSeriesSplit) path."""
import numpy as np

from mlframe.feature_selection._cv_splitters import TimestampOrderedSplit


def test_grouped_path_forwards_gap_embargo():
    n_groups, per = 20, 5
    groups = np.repeat(np.arange(n_groups), per)
    ts = np.arange(len(groups))
    gap = 2
    folds = list(TimestampOrderedSplit(n_splits=3, timestamps=ts, gap=gap).split(groups=groups))
    assert folds
    for tr, te in folds:
        train_groups, test_groups = set(groups[tr]), set(groups[te])
        assert not train_groups & test_groups
        assert min(test_groups) - max(train_groups) > gap
