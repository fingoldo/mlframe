"""Grouping rows by class made one full-length pass per class.

At 2M rows that is 0.16s at K=10 and 1.44s at K=200 -- linear in the class count, for an answer one sort
gives. The subtlety is that sorting the labels DIRECTLY is slower than the loop it replaces: numpy
radix-sorts narrow integers but falls back to a comparison sort on int64, where a stable argsort of 2M
labels costs 1.36s. Narrowing the sort key is what turns this from a wash into a win.

The sample itself must not move: these indices choose which rows the one-vs-rest curves are drawn from.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.multiclass import _narrowed, _stratified_subsample

CAP = 20_000


def _reference(y_pos: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """The per-class-pass implementation this replaces, kept as the identity oracle."""
    n = y_pos.shape[0]
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    frac = cap / n
    out = []
    for c in np.unique(y_pos):
        idx_c = np.flatnonzero(y_pos == c)
        take = min(max(1, round(len(idx_c) * frac)), len(idx_c))
        out.append(rng.choice(idx_c, size=take, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


@pytest.mark.parametrize("k", [2, 10, 57, 200])
def test_the_drawn_sample_is_unchanged(k):
    """Not "statistically equivalent" -- the same rows, because the RNG sees the same inputs in the same order."""
    y = np.random.default_rng(0).integers(0, k, 200_000)
    assert np.array_equal(_stratified_subsample(y, CAP), _reference(y, CAP)), f"the subsample moved at K={k}"


def test_an_imbalanced_column_keeps_every_class():
    """Proportional allocation with a floor of one: a rare class must not be sampled out of the curves."""
    y = np.zeros(100_000, dtype=np.int64)
    y[:5] = 1  # five rows of a second class
    idx = _stratified_subsample(y, CAP)
    assert set(np.unique(y[idx]).tolist()) == {0, 1}, "the rare class vanished from the subsample"


def test_a_small_column_is_returned_whole():
    """Under the cap there is nothing to choose between, so every row is kept."""
    y = np.random.default_rng(0).integers(0, 4, CAP // 2)
    assert np.array_equal(_stratified_subsample(y, CAP), np.arange(y.size))


@pytest.mark.parametrize(
    "values,expected",
    [
        (np.arange(10, dtype=np.int64), np.int16),
        (np.array([0, 40_000], dtype=np.int64), np.int32),
        (np.array([0, 2**40], dtype=np.int64), np.int64),
    ],
    ids=["small", "over-int16", "over-int32"],
)
def test_the_sort_key_is_narrowed_only_when_it_fits(values, expected):
    """An inexact cast would reorder the groups and move the sample; the range decides, not the class count."""
    assert _narrowed(values).dtype == expected


def test_narrowing_leaves_a_non_integer_key_alone():
    """Guard: the labels are normally positional codes, but the helper must not mangle anything else."""
    floats = np.array([0.5, 1.5, 2.5])
    assert _narrowed(floats).dtype == floats.dtype


def test_the_subsample_narrows_its_sort_key(monkeypatch):
    """The mechanism, not the wall clock: sorting the labels directly is SLOWER than the loop it replaces.

    Timing alone cannot pin this -- at moderate n an int64 argsort is fast enough to look fine -- so what is
    asserted is that the narrowing actually happens on the labels, which is where the win comes from.
    """
    import mlframe.reporting.charts.multiclass as mc

    seen = []
    real = mc._narrowed

    def spy(values):
        """Record what the subsample hands to the narrower."""
        seen.append(np.asarray(values).dtype)
        return real(values)

    monkeypatch.setattr(mc, "_narrowed", spy)
    y = np.random.default_rng(0).integers(0, 50, 100_000)
    mc._stratified_subsample(y, CAP)
    assert seen, "the sort key was not narrowed at all; on int64 labels that is slower than the loop this replaced"


def test_it_makes_one_pass_regardless_of_the_class_count():
    """The property, counted rather than timed.

    A wall-clock comparison against the per-class loop is not safe in a suite that runs under ``-n 4``:
    the two implementations contend for the same cores and the comparison flips. What the fix changed is
    the number of full-length passes -- one grouped sort instead of one mask per class -- and that is
    observable directly.
    """
    import mlframe.reporting.charts.multiclass as mc

    sorts = {"n": 0}
    real = mc.np.argsort

    def counted(a, **kw):
        """Count the sorts the subsample performs."""
        sorts["n"] += 1
        return real(a, **kw)

    for k in (3, 200):
        y = np.random.default_rng(0).integers(0, k, 60_000)
        sorts["n"] = 0
        mc.np.argsort = counted
        try:
            _stratified_subsample(y, CAP)
        finally:
            mc.np.argsort = real
        assert sorts["n"] == 1, f"grouping {k} classes took {sorts['n']} sorts; it must take exactly one whatever the count"
