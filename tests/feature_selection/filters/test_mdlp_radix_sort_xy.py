"""``mdlp_bin_edges`` of a large float32 column sorts ``(x, y)`` by a stable radix argsort and one gather; its edges must equal those of the ``np.argsort`` route.

The MDLP split search reads the sorted ``x`` and the label counts per distinct ``x``. The radix sort keeps equal ``x`` in input order (``np.argsort`` leaves them in an unspecified one; ``-0.0`` sorts before ``+0.0`` here), so the sorted ``x`` is identical and the edges must be too:
checked for a continuous column, columns with few and many ties, a column with zeros of both signs, a target with 2 and 7 classes, in the default validated mode and in ``fast_mode``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import supervised_binning as sb
from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


def _column(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """A float32 column of the given shape."""
    if kind == "continuous":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "few_levels":
        return rng.integers(0, 12, n).astype(np.float32)
    if kind == "many_ties":
        return rng.integers(0, 3000, n).astype(np.float32)
    x = np.where(rng.random(n) < 0.3, np.float32(0.0), rng.standard_normal(n).astype(np.float32))
    x[rng.random(n) < 0.1] = np.float32(-0.0)
    return x


@pytest.mark.parametrize("n", [4096, 6000, 30_000])
@pytest.mark.parametrize("kind", ["continuous", "few_levels", "many_ties", "zeros"])
@pytest.mark.parametrize("n_classes", [2, 7])
@pytest.mark.parametrize("fast_mode", [False, True])
def test_radix_route_gives_the_edges_of_the_argsort_route(n, kind, n_classes, fast_mode, monkeypatch):
    """Edges of the radix route equal the numpy route's for several column shapes, class counts and both modes."""
    rng = np.random.default_rng(n + n_classes + len(kind))
    x = _column(kind, n, rng)
    y = np.where(rng.random(n) < 0.6, (x > np.float32(0.2)).astype(np.int64) % n_classes, rng.integers(0, n_classes, n))
    fast = sb.mdlp_bin_edges(x, y, fast_mode=fast_mode)
    monkeypatch.setattr(sb, "sort_xy", lambda a, b: None)
    slow = sb.mdlp_bin_edges(x, y, fast_mode=fast_mode)
    assert np.array_equal(fast, slow), (fast, slow)


@pytest.mark.parametrize("n", [4096, 20_000])
@pytest.mark.parametrize("kind", ["continuous", "many_ties", "zeros"])
def test_sort_xy_sorts_x_and_carries_y_with_its_value(n, kind):
    """The sorted x is numpy's (as float64, -0.0 equal to +0.0) and every (value, label) pair of the input appears exactly once: equal values may come in another order, never with another label."""
    rng = np.random.default_rng(n + len(kind))
    x = _column(kind, n, rng)
    y = rng.integers(0, 5, n).astype(np.int64)
    got = qr.sort_xy(x, y)
    assert got is not None
    assert np.array_equal(got[0], np.sort(x).astype(np.float64))
    want_pairs = np.lexsort((y, x.astype(np.float64) + 0.0))
    got_pairs = np.lexsort((got[1], got[0] + 0.0))
    assert np.array_equal(got[0][got_pairs] + 0.0, x.astype(np.float64)[want_pairs] + 0.0)
    assert np.array_equal(got[1][got_pairs], y[want_pairs])
    stable = np.argsort(x, kind="stable") if kind != "zeros" else None
    if stable is not None:
        assert np.array_equal(got[1], y[stable])  # without signed zeros the order of ties is the stable one


def test_other_dtypes_and_small_inputs_return_none():
    """float64 ``x``, float label arrays, 2-D inputs, mismatched lengths and vectors under the switch-over size are left to numpy."""
    rng = np.random.default_rng(1)
    x32 = rng.standard_normal(10_000).astype(np.float32)
    yi = rng.integers(0, 3, 10_000)
    assert qr.sort_xy(x32.astype(np.float64), yi) is None
    assert qr.sort_xy(x32, yi.astype(np.float64)) is None
    assert qr.sort_xy(x32[:100], yi[:100]) is None
    assert qr.sort_xy(x32, yi[:-1]) is None
    assert qr.sort_xy(x32.reshape(100, 100), yi.reshape(100, 100)) is None
