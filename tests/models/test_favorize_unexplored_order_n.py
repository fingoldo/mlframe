"""MODELS-5 regression test: favorize_unexplored's order parameter must actually affect behavior.

The bug (fixed): the docstring promised favorizing "combinations of order N" not yet chosen in trials,
but the function body never referenced ``order`` at all -- only single categorical values (order=1
semantics) were ever favorized regardless of what ``order`` was passed. Fixed to score every size-order
combination of cat_features, favorizing co-occurrences not yet seen together even when each individual
value already appeared (with other companions) in trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.models.tuning import favorize_unexplored

pytestmark = pytest.mark.fast


def test_order_2_favorizes_novel_pair_even_when_each_value_individually_seen():
    """With order=2, a candidate whose (a, b) pair was never seen together must be favorized, even
    though `a` and `b` individually both already appear in trials (just never paired this way)."""
    trials = pd.DataFrame({"a": ["x", "y"], "b": ["y", "x"]})  # (x,y) and (y,x) seen; (x,x)/(y,y) never
    cands = [{"a": "x", "b": "x"}, {"a": "x", "b": "y"}]  # first is a novel pair, second was seen
    probs = np.ones(2) / 2

    favorize_unexplored(cands, probs, trials, ["a", "b"], order=2)
    assert probs[0] > probs[1], "the novel (x,x) pair should be favorized over the already-seen (x,y) pair"


def test_order_1_does_not_favorize_a_seen_pair_of_individually_novel_values():
    """Sanity contrast: with order=1, favorization is purely per-value, so a pair whose individual values
    are both already common gets no boost even if this exact pairing is new -- order=2 is what changes that."""
    trials = pd.DataFrame({"a": ["x", "y"], "b": ["y", "x"]})
    cands = [{"a": "x", "b": "x"}]
    probs = np.ones(1)
    favorize_unexplored(cands, probs, trials, ["a", "b"], order=1)
    # x is already in already_sampled["a"] and already_sampled["b"], so no field is individually novel.
    assert probs[0] == 1.0


def test_order_exceeding_cat_features_count_is_clamped_not_an_error():
    """order larger than len(cat_features) is clamped to len(cat_features), not an error."""
    trials = pd.DataFrame({"a": ["x"], "b": ["y"]})
    cands = [{"a": "z", "b": "w"}]
    probs = np.ones(1)
    favorize_unexplored(cands, probs, trials, ["a", "b"], order=5)
    assert probs[0] > 0


def test_order_below_one_raises():
    """order=0 or negative must raise, not silently misbehave."""
    trials = pd.DataFrame({"a": ["x"]})
    with pytest.raises(ValueError):
        favorize_unexplored([{"a": "x"}], np.ones(1), trials, ["a"], order=0)
