"""``_heldout_smooth_r2_fast`` ranks its column with the compiled average-rank kernel and returns the value the scipy ranks give."""

import numpy as np
from scipy.stats import rankdata

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import _avg_ranks_inplace
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as extra


def test_compiled_average_ranks_equal_scipy():
    """Both float widths, heavy ties and signed zeros: the ranks are scipy's ``rankdata(method="average")`` exactly."""
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(30, 3000))
        x = rng.standard_normal(n).astype(np.float32 if trial % 2 else np.float64)
        if trial % 3 == 0:
            x = np.round(x, 1)
        if trial % 5 == 0:
            x[:3] = [0.0, -0.0, 0.0]
        ranks = np.empty(n)
        _avg_ranks_inplace(np.ascontiguousarray(x), n, np.empty(n, dtype=np.int64), ranks)
        assert np.array_equal(ranks, rankdata(x, method="average")), trial


def test_r2_of_a_monotone_signal_is_high_and_of_noise_is_low():
    """A column that drives the target through a smooth monotone map scores a high held-out R2, an unrelated one about zero, a constant one exactly zero."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(6000)
    y = np.tanh(x) + 0.1 * rng.standard_normal(6000)
    prep = extra._heldout_smooth_r2_prep(y)
    assert extra._heldout_smooth_r2_fast(x.astype(np.float32), prep) > 0.9
    assert abs(extra._heldout_smooth_r2_fast(rng.standard_normal(6000).astype(np.float32), prep)) < 0.02
    assert extra._heldout_smooth_r2_fast(np.ones(6000, dtype=np.float32), prep) == 0.0
