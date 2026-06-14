"""Instance-vs-alias routing parity for model-identity gates in the training core.

mlframe accepts a model as a string alias ("cb", "mlp", ...) OR as an estimator
INSTANCE (``CatBoostClassifier()``). Gates that identify a model by a bare
``str(m).lower()`` / name-tuple membership test silently mis-route the instance
(which stringifies to ``"<catboost...object at 0x..>"``). These tests pin the
fixed behaviour: alias and instance route identically through the strategy
registry. They FAIL on the pre-fix name-only checks.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_is_catboost_model_alias_and_instance_parity():
    from mlframe.training.strategies import is_catboost_model

    cb = pytest.importorskip("catboost")
    inst = cb.CatBoostClassifier()
    assert is_catboost_model("cb") is True
    assert is_catboost_model("catboost") is True
    # The load-bearing case: an estimator INSTANCE must route the same as the alias.
    assert is_catboost_model(inst) is True
    assert is_catboost_model(("mycb", inst)) is True
    # Negatives stay negative.
    assert is_catboost_model("lgb") is False
    from sklearn.linear_model import LogisticRegression

    assert is_catboost_model(LogisticRegression()) is False


def test_is_neural_model_alias_and_instance_parity():
    from mlframe.training.strategies import (
        get_strategy,
        is_neural_model,
    )

    for alias in ("mlp", "lstm", "gru", "rnn", "transformer", "ngb"):
        assert is_neural_model(alias) is True, alias
    assert is_neural_model("cb") is False
    # Any estimator the registry classes as a neural/recurrent strategy must route True;
    # mirror that by asserting isinstance-consistency with get_strategy.
    from mlframe.training.strategies.neural import (
        NeuralNetStrategy,
        RecurrentModelStrategy,
    )

    assert isinstance(get_strategy("mlp"), NeuralNetStrategy)
    assert isinstance(get_strategy("lstm"), RecurrentModelStrategy)


def test_cb_gpu_probe_fires_for_instance_like_alias():
    """The CB-GPU task-type probe must run for a CatBoost INSTANCE in ``enabled_models``,
    not only for the ``"cb"`` alias. Pre-fix the ``str(m).lower() in ("cb","catboost")``
    gate missed the instance, silently forcing CB to CPU on a GPU host."""
    from mlframe.training.helpers import get_training_configs

    cb = pytest.importorskip("catboost")
    inst = cb.CatBoostClassifier()

    def _run(enabled):
        # has_gpu=True triggers the probe only when CB is judged in-scope.
        with mock.patch("mlframe.training.cb._cb_gpu_usable", return_value=False) as probe:
            get_training_configs(iterations=10, has_gpu=True, enabled_models=enabled)
        return probe.call_count

    alias_calls = _run(["cb"])
    inst_calls = _run([inst])
    tuple_calls = _run([("c", inst)])
    assert alias_calls >= 1
    # The parity assertion: the instance must trigger the probe exactly as the alias does.
    assert inst_calls == alias_calls
    assert tuple_calls == alias_calls


def test_cb_gpu_probe_skipped_for_non_cb_suite():
    """Non-CB suite must NOT pay the probe -- guards against over-broad routing."""
    from mlframe.training.helpers import get_training_configs

    with mock.patch("mlframe.training.cb._cb_gpu_usable", return_value=False) as probe:
        get_training_configs(iterations=10, has_gpu=True, enabled_models=["lgb"])
    assert probe.call_count == 0


def test_fit_pipeline_has_cb_detects_instance():
    """``_phase_fit_pipeline`` CB-native auto-flip + ordinal warning gate keys off a CB
    presence check; an instance must be detected the same as the alias."""
    from mlframe.training.strategies import is_catboost_model

    cb = pytest.importorskip("catboost")
    inst = cb.CatBoostClassifier()
    # Mirror the exact in-prod predicate (post-fix).
    assert any(is_catboost_model(m) for m in [inst])
    assert any(is_catboost_model(m) for m in ["cb"])
    assert not any(is_catboost_model(m) for m in ["lgb"])
