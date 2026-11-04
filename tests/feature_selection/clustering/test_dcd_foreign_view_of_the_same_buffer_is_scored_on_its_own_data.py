"""A different column block cut from the same buffer as the state's matrix is foreign data: it must not be served the state's cached scores."""
import numpy as np

from mlframe.feature_selection.filters._dynamic_cluster_discovery import make_dcd_state, pair_su


def test_other_column_block_of_the_same_buffer_gets_its_own_su():
    """Columns 0/1 of the state's block are independent (SU ~ 0); columns 0/1 of the other block of the same buffer are identical (SU ~ 1)."""
    rng = np.random.default_rng(0)
    n = 2000
    own = rng.integers(0, 4, (n, 3)).astype(np.int32)
    same = rng.integers(0, 4, n).astype(np.int32)
    other = np.column_stack([same, same, rng.integers(0, 4, n)]).astype(np.int32)
    big = np.ascontiguousarray(np.hstack([own, other]))
    own_view, other_view = big[:, :3], big[:, 3:]
    nb = np.array([4, 4, 4], dtype=np.int64)
    state = make_dcd_state(X_raw=None, factors_data=own_view, factors_nbins=nb, cols=["a", "b", "c"], nbins=nb, target_indices=np.array([2], dtype=np.int64), distance="sotoca_pla")
    su_own = pair_su(state, 0, 1, entropy_cache=None, factors_data=own_view, factors_nbins=nb)
    su_other = pair_su(state, 0, 1, entropy_cache=None, factors_data=other_view, factors_nbins=nb)
    assert su_own < 0.2 and su_other > 0.9, (su_own, su_other)
