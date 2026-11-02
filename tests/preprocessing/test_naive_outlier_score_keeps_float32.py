"""compute_naive_outlier_score scores float32 inputs without a float64 copy and gives the same result (tracker 2026-11-02 wave 16, MEM16-9)."""

import numpy as np

from mlframe.preprocessing import outliers


def test_float32_input_matches_the_float64_result(monkeypatch):
    """The comparison against train min/max is exact either way; the kernels must receive the float32 arrays themselves."""
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(400, 6)).astype(np.float32)
    te = (rng.normal(size=(100, 6)) * 1.5).astype(np.float32)
    seen = []
    real = outliers.count_num_outofranges
    monkeypatch.setattr(outliers, "count_num_outofranges", lambda x, mins, maxs: seen.append(x) or real(x, mins, maxs))
    got = outliers.compute_naive_outlier_score(tr, te)
    assert seen[0].dtype == np.float32 and np.shares_memory(seen[0], te)
    np.testing.assert_array_equal(got, outliers.compute_naive_outlier_score(tr.astype(np.float64), te.astype(np.float64)))


def test_integer_input_is_still_cast():
    """Non-floating inputs keep working."""
    got = outliers.compute_naive_outlier_score(np.arange(20).reshape(10, 2), np.array([[100, 0], [3, 4]]))
    np.testing.assert_array_equal(got, [1.0, 0.0])
