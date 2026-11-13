"""``_present_classes`` equals ``np.unique`` for any label vector and takes the count route only for a long non-negative integer vector with a small largest label.

The validated MDLP recursion asked ``np.unique`` of the 100000 root labels for every binned column (a sort, 1.5 ms); a count over the label range gives the same sorted distinct labels in one pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _mdlp_validated_split as vs


@pytest.mark.parametrize("n", [100, 4095, 4096, 50_000])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8])
@pytest.mark.parametrize("classes", [1, 2, 7, 60])
def test_matches_numpy_for_non_negative_integer_labels(n, dtype, classes):
    """Sorted distinct labels equal numpy's, including vectors with gaps in the label range and a single class."""
    rng = np.random.default_rng(n + classes)
    y = (rng.integers(0, classes, n) * 2 + (1 if classes > 1 else 0)).astype(dtype) if classes < 60 else rng.integers(0, classes, n).astype(dtype)
    np.testing.assert_array_equal(vs._present_classes(y), np.unique(y))


@pytest.mark.parametrize("make", ["negative", "huge", "float", "bool", "empty", "two_d_view"])
def test_other_inputs_take_numpy(make):
    """Negative labels, a huge largest label, floats, bools, an empty vector and a strided view are numpy's own result."""
    rng = np.random.default_rng(3)
    n = 20_000
    if make == "negative":
        y = rng.integers(-3, 4, n).astype(np.int64)
    elif make == "huge":
        y = rng.integers(0, 3, n).astype(np.int64)
        y[5] = 10**9
    elif make == "float":
        y = rng.integers(0, 4, n).astype(np.float64)
    elif make == "bool":
        y = rng.random(n) < 0.4
    elif make == "empty":
        y = np.empty(0, dtype=np.int64)
    else:
        y = rng.integers(0, 5, (n, 3)).astype(np.int64)[:, 1]
    np.testing.assert_array_equal(vs._present_classes(y), np.unique(y))


def test_the_count_route_is_taken_for_a_long_small_label_vector(monkeypatch):
    """A long non-negative integer vector never reaches ``np.unique`` (the sort), a negative one does."""
    calls = []
    real = np.unique
    monkeypatch.setattr(vs.np, "unique", lambda *a, **k: calls.append(1) or real(*a, **k))
    vs._present_classes(np.random.default_rng(0).integers(0, 3, 10_000).astype(np.int64))
    assert calls == []
    vs._present_classes(np.random.default_rng(0).integers(-1, 3, 10_000).astype(np.int64))
    assert calls == [1]
