"""HybridSelector turns a RandomState or Generator into an integer seed instead of raising TypeError (tracker 2026-11-02 wave 14, SKL14-9)."""

import numpy as np

from mlframe.feature_selection.hybrid_selector import _seed_int


def test_seed_int_accepts_every_random_state_flavour():
    """int(RandomState) raised TypeError; None keeps meaning seed 0."""
    assert _seed_int(None) == 0
    assert _seed_int(7) == 7
    assert isinstance(_seed_int(np.random.RandomState(0)), int)
    assert isinstance(_seed_int(np.random.default_rng(0)), int)
    assert _seed_int(np.random.RandomState(0)) == _seed_int(np.random.RandomState(0))
