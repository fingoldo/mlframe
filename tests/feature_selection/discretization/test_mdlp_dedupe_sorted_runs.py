"""The linear run pass of ``_dedupe_xy`` on a sorted column returns what the lexsort path returns."""

import numpy as np

from mlframe.feature_selection.filters._mdlp_validated_split import _dedupe_xy


def _lexsort_reference(x, y):
    """The lexsort path of ``_dedupe_xy``: the input arrays when no pair repeats, else the distinct pairs in (x, y) order."""
    order = np.lexsort((y, x))
    xs, ys = x[order], y[order]
    keep = np.empty(xs.size, dtype=bool)
    keep[0] = True
    keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    return (x, y) if keep.all() else (xs[keep], ys[keep])


def test_sorted_runs_equal_the_lexsort_path():
    """Sorted columns with ties at several roundings and label counts: same values, order and dtype as the lexsort path."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 2000))
        x = np.sort(np.round(rng.normal(size=n), int(rng.integers(0, 3))))
        y = rng.integers(0, int(rng.integers(2, 6)), n).astype(np.int64)
        got, ref = _dedupe_xy(x, y), _lexsort_reference(x, y)
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1]) and got[1].dtype == ref[1].dtype


def test_ties_without_repeated_pairs_keep_the_input_arrays():
    """Tied x with distinct labels inside every run: nothing repeats, so the input arrays come back as they are."""
    x = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
    y = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    got_x, got_y = _dedupe_xy(x, y)
    assert got_x is x and got_y is y
