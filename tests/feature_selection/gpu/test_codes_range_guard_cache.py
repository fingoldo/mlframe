"""The range guard checks a device code array once and keeps raising for arrays that are out of range."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_batched_mi_guards as g


def test_second_check_of_the_same_array_does_not_reduce_again(monkeypatch):
    """A valid array passes for any larger K, and the second call reads the cache (no min / max launched)."""
    a = cp.asarray(np.array([0, 1, 2, 3, 1], dtype=np.int64))
    g._assert_codes_in_range(a, 4, "t")
    calls = []
    real = cp.stack
    monkeypatch.setattr(cp, "stack", lambda *x, **k: (calls.append(1), real(*x, **k))[1])
    g._assert_codes_in_range(a, 10, "t")
    assert not calls


def test_out_of_range_still_raises_from_the_cache():
    """The cached range is compared with each call's K: too small a K raises, and a negative code raises."""
    a = cp.asarray(np.array([0, 5], dtype=np.int64))
    g._assert_codes_in_range(a, 6, "t")
    with pytest.raises(ValueError):
        g._assert_codes_in_range(a, 5, "t")
    with pytest.raises(ValueError):
        g._assert_codes_in_range(cp.asarray(np.array([-1, 2], dtype=np.int64)), 3, "t")
