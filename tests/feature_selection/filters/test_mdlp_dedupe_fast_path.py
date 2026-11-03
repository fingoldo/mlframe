"""``_dedupe_xy`` returns what the full lexsort dedup returns, with or without the strictly-increasing fast path."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_validated_split import _dedupe_xy


def _reference(x, y):
    """The lexsort dedup without any fast path."""
    order = np.lexsort((y, x))
    xs, ys = x[order], y[order]
    if xs.size <= 1:
        return x, y
    keep = np.empty(xs.size, dtype=bool)
    keep[0] = True
    keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    return (x, y) if keep.all() else (xs[keep], ys[keep])


def _cases():
    """Sorted distinct, sorted with ties and duplicate pairs, unsorted, NaN, and tiny inputs."""
    rng = np.random.default_rng(0)
    xs = np.sort(rng.standard_normal(500))
    tied = np.sort(np.round(rng.standard_normal(500), 1))
    out = [
        (xs, rng.integers(0, 3, 500)),
        (tied, rng.integers(0, 2, 500)),
        (rng.standard_normal(300), rng.integers(0, 2, 300)),
        (np.array([1.0, np.nan, 2.0]), np.array([0, 1, 0])),
        (np.array([3.0]), np.array([1])),
        (np.array([1.0, 1.0]), np.array([0, 0])),
    ]
    return out


@pytest.mark.parametrize("x,y", _cases())
def test_same_output_as_full_lexsort(x, y):
    """Same arrays (values and order) as the reference."""
    gx, gy = _dedupe_xy(x, y)
    wx, wy = _reference(x, y)
    assert np.array_equal(gx, wx, equal_nan=True) and np.array_equal(gy, wy)
