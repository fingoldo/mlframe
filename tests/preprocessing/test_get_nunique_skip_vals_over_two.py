"""PREPROCESSING-9 regression test: _get_nunique's float fast path must not silently drop skip_vals
beyond the first 2.

The bug (fixed): the njit fast-path kernel only ever extracts skip_vals[0]/skip_vals[1] -- any 3rd+
element of a longer skip_vals tuple was silently ignored (never excluded from the count), diverging
silently from the np.unique fallback path (non-float dtype), which supports arbitrary-length skip_vals.
Fixed by raising ValueError instead of silently miscounting.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.preprocessing.cleaning import _get_nunique

pytestmark = pytest.mark.fast


def test_float_fast_path_raises_on_more_than_two_skip_vals():
    """3+ skip_vals on a float array must raise, not silently drop the extras."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    with pytest.raises(ValueError, match="at most 2"):
        _get_nunique(vals, skip_vals=(1.0, 2.0, 3.0))


def test_float_fast_path_matches_object_fallback_for_two_skip_vals():
    """Sanity: 2 skip_vals still works and matches the np.unique-based fallback semantics."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    result = _get_nunique(vals, skip_vals=(1.0, 2.0))
    assert result == 2  # {3.0, 4.0} remain


def test_object_dtype_fallback_supports_more_than_two_skip_vals():
    """The np.unique fallback path (non-float dtype) genuinely supports arbitrary-length skip_vals --
    confirms the fast-path guard above isn't just a redundant, overly-strict restriction."""
    vals = np.array(["a", "b", "c", "d", "e"], dtype=object)
    result = _get_nunique(vals, skip_vals=("a", "b", "c"))
    assert result == 2  # {"d", "e"} remain
