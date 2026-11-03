"""The row matrix the raw-MI noise floor hashes and scores equals stacking its columns one by one, for strided, contiguous and mixed-dtype frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._unified_fe_gate import _column_values, _numeric_rows, raw_mi_noise_floor


def _frames():
    """A frame built from a C-ordered matrix (strided columns) with NaN, the same frame with an int8 column added, and a column-by-column frame."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((5000, 30)).astype(np.float32)
    base[::17, 3] = np.nan
    strided = pd.DataFrame(base, columns=[f"c{i}" for i in range(30)])
    mixed = strided.copy()
    mixed["k"] = rng.integers(0, 5, 5000).astype(np.int8)
    by_column = pd.DataFrame({f"c{i}": rng.standard_normal(5000) for i in range(12)})
    return [strided, mixed, by_column]


@pytest.mark.parametrize("frame", _frames(), ids=["strided", "mixed", "by_column"])
def test_rows_equal_column_by_column_stack(frame):
    """Bit-equal (NaN included) to ``np.stack`` of the per-column values at the fit's float dtype."""
    cols = list(frame.columns)
    want = np.stack([_column_values(frame, c, _crit_np_dtype()) for c in cols])
    got = _numeric_rows(frame, cols)
    assert got.dtype == want.dtype and got.flags.c_contiguous
    assert np.array_equal(got, want, equal_nan=True)


def test_floor_is_stable_across_memo_hits():
    """The second (memo-hit) call returns the first call's floor."""
    frame = _frames()[0]
    y = (np.nan_to_num(frame["c0"].to_numpy()) > 0).astype(int)
    first = raw_mi_noise_floor(frame, y)
    assert raw_mi_noise_floor(frame, y) == first
