"""pair_su / pair_vi: foreign data never shares the state's caches, and no matrix means unknown, not 0.0."""

import logging

import numpy as np

from mlframe.feature_selection.filters._dynamic_cluster_discovery import DCDState
from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_metrics import pair_su, pair_vi


def _state(fd, fn):
    st = DCDState()
    st.factors_data = fd
    st.factors_nbins = fn
    return st


def test_foreign_data_does_not_poison_the_states_cache():
    """Scoring an override matrix used to write its SU into the cache under the same pair key."""
    rng = np.random.default_rng(0)
    own = rng.integers(0, 4, size=(500, 2)).astype(np.int32)  # independent columns
    twin = np.column_stack([own[:, 0], own[:, 0]]).astype(np.int32)  # identical columns
    nb = np.array([4, 4], dtype=np.int64)
    st = _state(own, nb)
    su_twin = pair_su(st, 0, 1, factors_data=twin, factors_nbins=nb)
    su_own = pair_su(st, 0, 1)
    assert su_twin > 0.99, "identical columns are fully redundant"
    assert su_own < 0.1, "the state's own independent columns must not be served the override's SU"


def test_no_matrix_is_unknown_not_independent(caplog):
    st = _state(None, None)
    with caplog.at_level(logging.WARNING):
        su, vi = pair_su(st, 0, 1), pair_vi(st, 0, 1)
    assert np.isnan(su) and np.isnan(vi), "0.0 would claim 'independent' (SU) or 'equivalent' (VI) - measurements nobody made"
    assert any("no factors matrix" in r.getMessage() for r in caplog.records)


def test_the_states_own_arrays_passed_back_in_stay_on_the_cached_path():
    """DCD passes the caller's original nbins list; the state holds np.asarray of it. That is the same data."""
    rng = np.random.default_rng(1)
    fd = rng.integers(0, 4, size=(200, 3)).astype(np.int32)
    nbins_list = [4, 4, 4]
    st = _state(fd, np.asarray(nbins_list))
    pair_su(st, 0, 1, factors_data=fd, factors_nbins=nbins_list)
    assert (0, 1) in st.pairwise_su_cache, "the state's own data must be cached, not routed through the scratch path"
