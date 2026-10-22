"""The conditional-twin pass drops a noisy copy that adds nothing, keeps a near-twin that does, and never touches a column without a correlated partner."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._hybrid_twin_prune import prune_conditional_twins


def _chain(seed, informative_bridge, n=3000):
    """``b = a + 0.6 e1``; ``c`` carries the independent component ``d``; ``y = (a + d > 0)``. With ``informative_bridge`` the column ``c`` is built from ``b`` (then ``b`` removes the ``e1`` noise from the estimate of ``d``)."""
    rng = np.random.default_rng(seed)
    a, e1, e2, d = (rng.normal(size=n) for _ in range(4))
    b = a + 0.6 * e1
    c = (b if informative_bridge else a) + 0.6 * e2 + d
    y = ((a + d + 0.4 * rng.normal(size=n)) > 0).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "n0": rng.normal(size=n)}), y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_redundant_noisy_copy_is_dropped(seed):
    """``b`` is a noisy copy of ``a`` and ``c`` does not use it: given ``a`` it adds nothing, and it goes."""
    X, y = _chain(seed, informative_bridge=False)
    kept = prune_conditional_twins(X, y, ["a", "b", "c"], {"a": 0.3, "b": 0.2, "c": 0.25})
    assert "b" not in kept and {"a", "c"} <= set(kept)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_twin_that_adds_information_stays(seed):
    """When ``c = b + noise + d`` the bridge ``b`` removes the ``e1`` noise from the estimate of ``d`` (AUC 0.938 with it, 0.913 without), so it is kept."""
    X, y = _chain(seed, informative_bridge=True)
    kept = prune_conditional_twins(X, y, ["a", "b", "c"], {"a": 0.3, "b": 0.2, "c": 0.25})
    assert "b" in kept


def test_columns_without_a_correlated_partner_are_never_tested():
    """Uncorrelated columns pass through untouched, in their order, even when one carries no signal."""
    X, y = _chain(0, informative_bridge=False)
    kept = prune_conditional_twins(X, y, ["c", "n0", "a"], {"a": 0.3, "c": 0.25, "n0": 0.0})
    assert kept == ["c", "n0", "a"]


def test_regression_target_drops_the_redundant_copy():
    """The same rule on a continuous target: a noisy copy of ``a`` adds nothing."""
    rng = np.random.default_rng(5)
    n = 3000
    a, e, d = (rng.normal(size=n) for _ in range(3))
    X = pd.DataFrame({"a": a, "b": a + 0.5 * e, "c": a + d})
    y = a + d + 0.3 * rng.normal(size=n)
    kept = prune_conditional_twins(X, y, ["a", "b", "c"], {"a": 0.3, "b": 0.1, "c": 0.2})
    assert "b" not in kept


def test_a_failure_leaves_the_selection_as_it_was():
    """A frame the pass cannot handle returns the selection unchanged."""
    kept = prune_conditional_twins(pd.DataFrame({"a": [1, 2]}), [0, 1], ["a", "zzz"], {})
    assert kept == ["a", "zzz"]


@pytest.mark.parametrize("seed", [0, 1])
def test_the_tree_model_check_also_drops_a_redundant_copy_and_keeps_an_informative_twin(seed):
    """With ``model_kind="hgb"`` the held-out fit is a gradient-boosted tree: the redundant copy still goes, the bridge that removes noise from the estimate of ``d`` still stays."""
    X, y = _chain(seed, informative_bridge=False)
    kept = prune_conditional_twins(X, y, ["a", "b", "c"], {"a": 0.3, "b": 0.2, "c": 0.25}, model_kind="hgb")
    assert "b" not in kept and {"a", "c"} <= set(kept)
    X, y = _chain(seed, informative_bridge=True)
    kept = prune_conditional_twins(X, y, ["a", "b", "c"], {"a": 0.3, "b": 0.2, "c": 0.25}, model_kind="hgb")
    assert "b" in kept
