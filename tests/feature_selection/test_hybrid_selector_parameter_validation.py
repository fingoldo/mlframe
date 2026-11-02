"""HybridSelector keeps its constructor arguments as given and rejects settings that silently changed the selection (tracker 2026-11-02, VAL6-4, VAL6-5)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.hybrid_selector import HybridSelector


def _data(n=120, seed=0):
    """A tiny frame with one informative column."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    return X, (X["a"] + 0.2 * rng.normal(size=n) > 0).astype(int)


def test_a_list_valued_tree_rich_ops_is_stored_as_given():
    """The constructor keeps the argument unchanged (no tuple copy), as the estimator contract expects."""
    ops = ["mul", "absd"]
    assert HybridSelector(tree_rich_ops=ops).tree_rich_ops is ops


def test_a_bare_string_is_one_operator_not_its_characters():
    """'absd' resolves to the single operator 'absd'."""
    assert HybridSelector(tree_rich_ops="absd")._resolved_tree_ops() == ("absd",)


@pytest.mark.parametrize("ops", [("mul", "nope"), "abs", ["zzz"]])
def test_unknown_operator_names_raise_in_fit(ops):
    """A typo is an error naming the bad operator, not a silent drop."""
    X, y = _data()
    with pytest.raises(ValueError, match="unknown tree_rich_ops"):
        HybridSelector(tree_rich_ops=ops).fit(X, y)


@pytest.mark.parametrize("vote", [0, -1, 1.5, True])
def test_vote_below_one_or_not_an_integer_raises_in_fit(vote):
    """vote=0 confirmed every cluster and turned the vote filter off; it and non-integers now raise."""
    X, y = _data()
    with pytest.raises(ValueError, match="vote must be an integer"):
        HybridSelector(vote=vote).fit(X, y)
