"""``score_pair_mi``'s plug-in estimator, and the absence of the dormant unique-y cache that used to wrap it.

``score_pair_mi`` / ``_score_plug_in`` have no caller in the per-pair MRMR loop, so the id(y)-keyed LRU that once
memoised ``np.unique(y)`` for them could never register a hit; it is gone, and the estimator calls ``np.unique``
directly.
"""

import numpy as np

from mlframe.feature_selection.filters import _mi_dispatch
from mlframe.feature_selection.filters._mi_dispatch import score_pair_mi


def test_the_dormant_unique_y_cache_is_gone():
    """A cache on a path with no caller only costs a lock, a weakref and the arrays it retains."""
    assert not hasattr(_mi_dispatch, "_UNIQ_Y_CACHE")
    assert not hasattr(_mi_dispatch, "_get_unique_y")


def test_score_pair_mi_plug_in_is_deterministic():
    """The plug-in estimator gives the same finite, non-negative MI on repeated calls with the same inputs."""
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y = rng.choice([0.0, 1.0], size=n)
    mi1 = score_pair_mi(x, y, estimator="plug_in")
    mi2 = score_pair_mi(x, y, estimator="plug_in")
    assert mi1 == mi2
    assert np.isfinite(mi1) and mi1 >= 0.0
