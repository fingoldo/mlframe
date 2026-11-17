"""A TimestampOrderedSplit that lost its bound groups to pickling says so instead of silently dropping group isolation."""

import logging
import pickle

import numpy as np

from mlframe.feature_selection._cv_splitters import TimestampOrderedSplit


def test_unpickled_splitter_warns_that_its_groups_are_gone(caplog):
    """Splitting after unpickle logs the lost groups."""
    n = 60
    sp = TimestampOrderedSplit(n_splits=3, groups=np.repeat(np.arange(12), 5))
    clone = pickle.loads(pickle.dumps(sp))  # nosec B301 -- round-trips an object this test just built
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        list(clone.split(np.zeros((n, 1))))
    assert any("groups were not pickled" in r.getMessage() for r in caplog.records)
