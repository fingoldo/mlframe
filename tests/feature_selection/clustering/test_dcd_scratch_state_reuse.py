"""Pair SU on an override matrix (the screen's row subsample) is computed once per pair: the scratch state and its caches persist across calls."""

import numpy as np

from mlframe.feature_selection.filters._dynamic_cluster_discovery import DCDState, pair_su
from mlframe.feature_selection.filters._dcd_pair_su_batch import pair_su_batch


def _state(n_full: int, n_cols: int, seed: int = 0):
    """A state that owns a full-size matrix, plus a row-subsample of it as the override."""
    rng = np.random.default_rng(seed)
    factors = rng.integers(0, 4, size=(n_full, n_cols)).astype(np.int32)
    factors[:, 1] = factors[:, 0]  # a redundant pair
    nbins = np.full(n_cols, 4, dtype=np.int64)
    state = DCDState(
        pool_pruned_mask=np.zeros(n_cols, dtype=bool),
        X_raw_ref=None,
        factors_data=factors,
        factors_nbins=nbins,
        cols=[f"c{i}" for i in range(n_cols)],
        nbins=nbins,
        target_indices=np.array([n_cols - 1], dtype=np.int64),
        quantization_method="quantile",
        quantization_nbins=4,
        quantization_dtype=np.int32,
        cluster_size_threshold=2,
        min_cluster_size=2,
        swap_gain_threshold=0.05,
        tau_cluster=0.5,
        swap_method="pca_pc1",
    )
    sub = np.ascontiguousarray(factors[: n_full // 2])
    return state, sub, nbins


def test_override_matrix_is_scored_once_per_pair_and_values_are_unchanged():
    """The batch warm-up and the later single calls share one cache; the SU values equal those of a fresh scratch state per call."""
    state, sub, nbins = _state(2000, 6)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    batch = pair_su_batch(state, pairs, factors_data=sub, factors_nbins=nbins)
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import _dcd_metrics as M

    scratch = next(iter(M._SCRATCH_BY_STATE[id(state)].values()))[0]
    computed_by_batch = scratch.n_su_calls
    assert computed_by_batch == len(pairs), "each pair is scored once by the batch"
    again = [pair_su(state, a, b, factors_data=sub, factors_nbins=nbins) for a, b in pairs]
    np.testing.assert_array_equal(batch, np.asarray(again))
    fresh = [pair_su(M._new_scratch_state(state, sub, nbins), a, b) for a, b in pairs]
    np.testing.assert_allclose(again, fresh, rtol=0, atol=1e-12)
    assert len(M._SCRATCH_BY_STATE[id(state)]) == 1, "one scratch state serves every call on the same override arrays"
    assert scratch.n_su_calls == computed_by_batch, "the single calls after the batch were cache hits"
    assert batch[0] > 0.99, "the redundant pair reads as redundant"
