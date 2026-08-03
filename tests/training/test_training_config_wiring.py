"""Regression test for the ``TrainingConfig`` sub-config wiring bug: ``TrainingConfig.linear_config`` /
``.behavior`` were declared but never consumed by ``train_mlframe_models_suite`` -- any value the user
populated on them was silently discarded (audit-2026-04-28 sweep failure mode, ``test_subconfig_wiring_parity``).
Fixed by threading them through the suite's new ``training_config=`` overlay parameter.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mlframe.training.configs import LinearModelConfig, TrainingBehaviorConfig, TrainingConfig
from mlframe.training.core import _main_train_suite as main_train_suite_module


def test_training_config_overlays_linear_and_behavior_when_kwargs_omitted(monkeypatch):
    """``training_config.linear_config`` / ``.behavior`` reach ``setup_configuration`` as
    ``linear_model_config`` / ``behavior_config`` when those explicit kwargs are left as None."""
    captured = {}

    def fake_setup_configuration(**kwargs):
        """Capture the received linear_model_config/behavior_config, then abort the suite call."""
        captured["linear_model_config"] = kwargs.get("linear_model_config")
        captured["behavior_config"] = kwargs.get("behavior_config")
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(main_train_suite_module.pr, "setup_configuration", fake_setup_configuration)

    training_config = TrainingConfig(
        target_name="y",
        model_name="m",
        linear_config=LinearModelConfig(model_type="ridge", alpha=3.5),
        behavior=TrainingBehaviorConfig(prefer_gpu_configs=False),
    )
    df = pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        main_train_suite_module.train_mlframe_models_suite(df, target_name="y", training_config=training_config)

    assert captured["linear_model_config"].alpha == 3.5
    assert captured["behavior_config"].prefer_gpu_configs is False


def test_explicit_kwarg_wins_over_training_config(monkeypatch):
    """An explicitly-passed ``linear_model_config`` / ``behavior_config`` overrides the ``training_config``
    overlay -- the overlay only fills in when the explicit kwarg is left as None."""
    captured = {}

    def fake_setup_configuration(**kwargs):
        """Capture the received linear_model_config/behavior_config, then abort the suite call."""
        captured["linear_model_config"] = kwargs.get("linear_model_config")
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(main_train_suite_module.pr, "setup_configuration", fake_setup_configuration)

    training_config = TrainingConfig(
        target_name="y",
        model_name="m",
        linear_config=LinearModelConfig(model_type="ridge", alpha=3.5),
    )
    explicit = LinearModelConfig(model_type="lasso", alpha=1.0)
    df = pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        main_train_suite_module.train_mlframe_models_suite(
            df, target_name="y", training_config=training_config, linear_model_config=explicit,
        )

    assert captured["linear_model_config"] is explicit


def test_training_config_defaults_to_none_and_is_a_pure_noop():
    """Without ``training_config=``, ``linear_model_config`` / ``behavior_config`` are unaffected
    (verified at the signature level: the new parameter defaults to None)."""
    import inspect

    sig = inspect.signature(main_train_suite_module.train_mlframe_models_suite)
    assert sig.parameters["training_config"].default is None
