"""The stability vote slices the frame once per fold for all its recipes, not once per recipe and fold.

Every recipe is replayed on the same held-out folds; slicing all the columns of the frame for each (recipe, fold) pair cost 14 ms each at 100000 rows. ``_fold_frame`` keeps the slice of a fold in the vote's own cache dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _fe_stability_vote as sv


def _frame(n=1000, k=6):
    """A small float frame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((n, k)), columns=[f"c{i}" for i in range(k)])


def test_a_fold_is_sliced_once_for_every_recipe(monkeypatch):
    """Five folds and twenty recipes: five slices, each equal to ``X.iloc[idx]``, the same object handed to every recipe."""
    X = _frame()
    folds = [np.arange(i, 1000, 5) for i in range(5)]
    calls = []
    real = sv._slice_fold
    monkeypatch.setattr(sv, "_slice_fold", lambda frame, idx: calls.append(1) or real(frame, idx))
    cache: dict = {}
    seen = {}
    for _recipe in range(20):
        for fold_no, idx in enumerate(folds):
            frame = sv._fold_frame(cache, fold_no, idx, X)
            assert seen.setdefault(fold_no, frame) is frame
            pd.testing.assert_frame_equal(frame, X.iloc[idx])
    assert len(calls) == 5


def test_another_index_array_for_the_same_fold_is_sliced_again(monkeypatch):
    """A fold number reused with a different index array (a new split) gets a fresh slice, never the stale one."""
    X = _frame()
    cache: dict = {}
    first = sv._fold_frame(cache, 0, np.arange(0, 500), X)
    second = sv._fold_frame(cache, 0, np.arange(500, 1000), X)
    assert first is not second
    pd.testing.assert_frame_equal(second, X.iloc[np.arange(500, 1000)])


def test_column_name_cache_entries_and_fold_entries_do_not_collide():
    """The vote's cache already holds column-name keys; the fold slices live under tuple keys and leave those untouched."""
    X = _frame()
    cache = {"c0": np.arange(3)}
    sv._fold_frame(cache, 0, np.arange(10), X)
    assert np.array_equal(cache["c0"], np.arange(3)) and ("__fold_frame__", 0) in cache
