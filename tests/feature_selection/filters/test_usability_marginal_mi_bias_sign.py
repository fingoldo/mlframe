"""The usability marginal MI and its batched GPU twin add the Miller-Madow term like the greedy CMI path (tracker 2026-11-02 wave 12, PAR12M-1, PAR12M-2)."""

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe_steps import marginal_mi_binned_fixed_y, precompute_marginal_y_terms
from mlframe.feature_selection.filters._numba_utils_usability import _marginal_mi_njit


def test_njit_marginal_mi_matches_the_greedy_path_and_is_near_zero_on_noise():
    """Independent 20x20 binned columns at n=2000: both implementations agree to 1e-12 and average near zero (the old sign gave ~0.18)."""
    rng = np.random.default_rng(0)
    n = 2000
    got, ref = [], []
    for _ in range(25):
        x = rng.integers(0, 20, n).astype(np.int64)
        y = rng.integers(0, 20, n).astype(np.int64)
        y_i, h_y, k_y = precompute_marginal_y_terms(y)
        ref.append(marginal_mi_binned_fixed_y(x, y_i, h_y, k_y))
        got.append(_marginal_mi_njit(x, 20, y_i, h_y, k_y, n))
    np.testing.assert_allclose(got, ref, atol=1e-12)
    assert np.mean(got) < 0.03
