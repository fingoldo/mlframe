"""The range guard checks a device code array on every call and raises for arrays that are out of range."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_batched_mi_guards as g


def test_second_check_of_the_same_array_reduces_again(monkeypatch):
    """Every check reads the array: a second call launches the min and max again, so a buffer rewritten in place is never judged by an old range."""
    a = cp.asarray(np.array([0, 1, 2, 3, 1], dtype=np.int64))
    g._assert_codes_in_range(a, 4, "t")
    calls = []
    real = cp.stack
    monkeypatch.setattr(cp, "stack", lambda *x, **k: (calls.append(1), real(*x, **k))[1])
    g._assert_codes_in_range(a, 10, "t")
    assert calls


def test_rewritten_buffer_at_the_same_pointer_is_checked_again():
    """The regression the cache allowed: the same device buffer rewritten with an out-of-range code under the same pointer must raise."""
    a = cp.asarray(np.array([0, 1], dtype=np.int64))
    ptr = int(a.data.ptr)
    g._assert_codes_in_range(a, 2, "t")
    a[:] = cp.asarray(np.array([0, 7], dtype=np.int64))
    assert int(a.data.ptr) == ptr
    with pytest.raises(ValueError):
        g._assert_codes_in_range(a, 2, "t")


def test_out_of_range_still_raises_from_the_cache():
    """The range is compared with each call's K: too small a K raises, and a negative code raises."""
    a = cp.asarray(np.array([0, 5], dtype=np.int64))
    g._assert_codes_in_range(a, 6, "t")
    with pytest.raises(ValueError):
        g._assert_codes_in_range(a, 5, "t")
    with pytest.raises(ValueError):
        g._assert_codes_in_range(cp.asarray(np.array([-1, 2], dtype=np.int64)), 3, "t")
