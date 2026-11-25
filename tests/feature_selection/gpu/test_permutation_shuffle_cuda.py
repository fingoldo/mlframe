"""The sort-free target shuffles (``_permutation_shuffle_cuda``): every row is a permutation of the labels, the permutations are uniform-looking by the statistics of a uniform permutation, and they are reproducible per seed."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _permutation_shuffle_cuda as shuf

if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)


def _images(n: int, nperm: int, seed: int = 0) -> np.ndarray:
    """``(nperm, n)`` int64: row ``k`` is the permutation ``pi_k`` itself (the labels are ``0 .. n - 1``)."""
    return cp.asnumpy(shuf.shuffled_rows(cp, cp.arange(n, dtype=cp.int64), nperm, seed))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 31, 32, 33, 100, 1009, 4096, 4097, 100000])
def test_every_row_is_a_permutation_of_the_labels(n):
    """Cycle-walking makes a bijection of ``0 .. n - 1`` for every n, below, at and above powers of two and of four."""
    rows = _images(n, 6, seed=n)
    assert rows.shape == (6, n)
    expected = np.arange(n)
    for row in rows:
        assert np.array_equal(np.sort(row), expected)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_the_labels_are_gathered_in_their_own_type(dtype):
    """The output rows hold the labels (a multiset equal to the input's), whatever the integer type."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 777).astype(dtype)
    out = cp.asnumpy(shuf.shuffled_rows(cp, cp.asarray(y), 5, 3))
    assert out.dtype == dtype
    for row in out:
        assert np.array_equal(np.sort(row), np.sort(y))


def test_the_rows_are_reproducible_per_seed_and_differ_between_rows_and_seeds():
    """The same seed gives the same matrix; another seed another one; the rows of one matrix are different permutations."""
    a, b, c = _images(500, 8, 1), _images(500, 8, 1), _images(500, 8, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len({tuple(r) for r in a}) == 8


def test_the_image_of_every_position_is_uniform():
    """For n = 40 and 40000 permutations, the counts of ``pi(i) = j`` follow the multinomial of a uniform permutation (chi-square over the 40 x 40 table)."""
    from scipy.stats import chi2

    n, k = 40, 40000
    rows = _images(n, k, seed=5)
    counts = np.zeros((n, n))
    for i in range(n):
        counts[i] = np.bincount(rows[:, i], minlength=n)
    stat = float(((counts - k / n) ** 2 / (k / n)).sum())
    p = float(chi2.sf(stat, n * (n - 1)))
    assert p > 1e-3, (stat, p)


def test_fixed_points_descents_and_successions_match_a_uniform_permutation():
    """Means over 20000 permutations of n = 60: fixed points 1, descents (n - 1) / 2, successions (n - 1) / n, each within five standard errors of the uniform values."""
    n, k = 60, 20000
    rows = _images(n, k, seed=11)
    fixed = (rows == np.arange(n)).sum(axis=1)
    descents = (rows[:, :-1] > rows[:, 1:]).sum(axis=1)
    successions = (rows[:, 1:] == rows[:, :-1] + 1).sum(axis=1)
    assert abs(fixed.mean() - 1.0) < 5 * 1.0 / np.sqrt(k)
    assert abs(descents.mean() - (n - 1) / 2) < 5 * np.sqrt((n + 1) / 12) / np.sqrt(k)
    assert abs(successions.mean() - (n - 1) / n) < 5 * 1.0 / np.sqrt(k)


def test_two_rows_are_not_related():
    """The correlation of the images of two rows over positions stays near zero (a shared structure between rows would show)."""
    rows = _images(5000, 64, seed=2).astype(np.float64)
    corr = np.corrcoef(rows)
    off = corr[~np.eye(64, dtype=bool)]
    assert np.abs(off).max() < 0.1
