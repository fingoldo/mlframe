"""The auto-tune fingerprint reads a float32 ndarray without a float64 twin of it (tracker 2026-11-02 wave 16, MEM16-10)."""

import tracemalloc

import numpy as np

from mlframe.feature_selection.wrappers._auto_tune import _array_column_signals


def test_signals_match_the_float64_result_and_allocate_less_than_a_twin():
    """20000 x 50 float32: the float64 copy alone is 8 MB; the signals must come out equal with peak extra allocation well below that."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 50)).astype(np.float32)
    y = X[:, 0] + rng.normal(size=20_000).astype(np.float32)
    ref = _array_column_signals(X.astype(np.float64), y, 50)
    tracemalloc.start()
    got = _array_column_signals(X, y, 50)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert got[0] == ref[0]
    np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-4)
    assert peak < 0.6 * X.shape[0] * X.shape[1] * 8
