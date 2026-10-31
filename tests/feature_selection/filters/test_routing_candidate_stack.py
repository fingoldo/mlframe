"""The routing candidate matrix equals ``np.column_stack(...).astype(dtype)`` for float64, float32 and mixed inputs."""

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._orthogonal_routing_fe import _stack_candidates


def test_stack_equals_column_stack_astype():
    """Bit-for-bit equality, including values that round differently in float32."""
    rng = np.random.default_rng(1)
    n = 1000
    values = [rng.normal(size=n) * 10.0 ** rng.integers(-6, 8) for _ in range(9)]
    values.append(rng.normal(size=n).astype(np.float32))
    values[2][5] = np.nan
    expected = np.column_stack(values).astype(_crit_np_dtype(), copy=False)
    got = _stack_candidates(values)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert np.array_equal(got.view(np.uint8), expected.view(np.uint8))
