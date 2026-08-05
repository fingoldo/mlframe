"""TRAINING_FEATURE_HANDLING_TARGETS-5 regression test: every string alias is_neural_model() recognizes
must also be a registered MODEL_STRATEGIES key routing to a neural/recurrent strategy.

The bug (fixed): is_neural_model() included the literal 'recurrent' in its alias set, but
MODEL_STRATEGIES had no 'recurrent' key -- get_strategy('recurrent') fell through to the
unknown-alias branch (warns + mis-routes to TreeModelStrategy) even though is_neural_model('recurrent')
reported True. Same bug class already fixed once in _helpers_training_configs.py.
"""

from __future__ import annotations

import warnings

import pytest

from mlframe.training.strategies import (
    NeuralNetStrategy,
    RecurrentModelStrategy,
    get_strategy,
    is_neural_model,
)

pytestmark = pytest.mark.fast

_NEURAL_ALIASES = ("mlp", "recurrent", "ngb", "lstm", "gru", "rnn", "transformer")


@pytest.mark.parametrize("alias", _NEURAL_ALIASES)
def test_is_neural_model_true_for_every_declared_alias(alias):
    """Sanity: every alias in the hardcoded set actually reports True."""
    assert is_neural_model(alias) is True


@pytest.mark.parametrize("alias", _NEURAL_ALIASES)
def test_get_strategy_does_not_warn_and_routes_to_a_neural_strategy(alias):
    """Every alias is_neural_model() recognizes must route to a real neural/recurrent strategy via
    get_strategy, without the "Unknown model" fallback warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strategy = get_strategy(alias)
    assert isinstance(
        strategy, (NeuralNetStrategy, RecurrentModelStrategy)
    ), f"get_strategy({alias!r}) should route to a neural/recurrent strategy, got {type(strategy).__name__}"


def test_recurrent_alias_specifically_in_model_strategies():
    """'recurrent' specifically must be a MODEL_STRATEGIES key (the exact bug reported)."""
    from mlframe.training.strategies import MODEL_STRATEGIES

    assert "recurrent" in MODEL_STRATEGIES
    assert isinstance(MODEL_STRATEGIES["recurrent"], RecurrentModelStrategy)
