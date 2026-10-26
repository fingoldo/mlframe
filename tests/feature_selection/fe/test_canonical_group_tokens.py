"""The vectorised token builder gives exactly the strings of the per-value ``canonical_group_token``, and the target-encoding key builder keeps its NaN sentinel."""
import numpy as np
import pytest

from mlframe.feature_selection.filters._internals import canonical_group_token, canonical_group_tokens
from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str
import pandas as pd


def _adversarial() -> np.ndarray:
    """Floats around every formatting edge: integral, tiny, huge, signed zero, non-finite, int64 limits."""
    rng = np.random.default_rng(0)
    special = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-5, 1e-300, 1.5e300, 1e15, 1e16, 2.0**53, 2.0**63, -(2.0**63), 9.0e18, 1.2e19, 1e22, np.nan, np.inf, -np.inf, 123456789.0, 0.30000000000000004]
    return np.concatenate([np.array(special), rng.normal(size=500) * 10.0 ** rng.integers(-8, 12, 500), np.round(rng.normal(size=200) * 50)])


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.uint32, np.int8, np.bool_])
def test_tokens_equal_the_per_value_function(dtype):
    """Every dtype: same strings, element for element."""
    values = _adversarial()
    if np.dtype(dtype).kind in "iu":
        values = np.round(values[np.isfinite(values) & (np.abs(values) < 100)]).clip(0, 100)
    elif np.dtype(dtype).kind == "b":
        values = values[:6] > 0
    arr = values.astype(dtype)
    want = np.array([canonical_group_token(v) for v in arr], dtype=object)
    got = canonical_group_tokens(arr)
    assert got.tolist() == want.tolist()


def test_column_to_str_keeps_the_nan_sentinel_for_floats():
    """A float column with NaN maps NaN to ``__nan__`` and integral floats to their int token."""
    out = _column_to_str(pd.Series([1.0, np.nan, 2.5, 1.0, np.nan]))
    assert out.tolist() == ["1", "__nan__", "2.5", "1", "__nan__"]
