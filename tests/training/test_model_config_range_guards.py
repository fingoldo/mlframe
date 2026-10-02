"""Range guards on the per-model config dataclasses.

Each test pins a previously-silent out-of-range value that propagated to the
backend (sklearn / LightGBM / XGB) and either produced a degenerate
no-error run (iterations=0 -> zero-tree booster predicting the init constant)
or surfaced as a confusing error deep inside fit. The guards reject them at
construction with a clear ValidationError, matching the established
ModelHyperparamsConfig contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlframe.training.configs import (
    LinearModelConfig,
    MLPConfig,
    ModelHyperparamsConfig,
    MultilabelDispatchConfig,
)
from mlframe.training._model_configs import validate_nested_mlp_kwargs

# ---- ModelHyperparamsConfig (the tree backends' shared knobs) --------------


def test_hyperparams_iterations_zero_raises():
    """Zero iterations trains zero trees; it is rejected at construction."""
    with pytest.raises(ValidationError):
        ModelHyperparamsConfig(iterations=0)


def test_hyperparams_learning_rate_negative_raises():
    """A negative learning rate is rejected at construction."""
    with pytest.raises(ValidationError):
        ModelHyperparamsConfig(learning_rate=-0.1)


def test_hyperparams_learning_rate_above_one_raises():
    """A learning rate above one is rejected at construction."""
    with pytest.raises(ValidationError):
        ModelHyperparamsConfig(learning_rate=5.0)


def test_hyperparams_early_stopping_zero_raises_and_none_disables():
    """0 is not an "auto" patience here (give it explicitly); None disables early stopping."""
    with pytest.raises(ValidationError):
        ModelHyperparamsConfig(early_stopping_rounds=0)
    assert ModelHyperparamsConfig(early_stopping_rounds=None).early_stopping_rounds is None


def test_hyperparams_valid_config_unchanged():
    """A valid configuration keeps its values."""
    cfg = ModelHyperparamsConfig(iterations=700, learning_rate=0.1)
    assert cfg.iterations == 700 and cfg.learning_rate == 0.1


# ---- MLPConfig / mlp_kwargs ------------------------------------------------


def test_mlp_kwargs_unknown_section_raises():
    """A misspelled section of the nested ``mlp_kwargs`` used to be ignored silently."""
    with pytest.raises(ValidationError, match="trainer_param"):
        validate_nested_mlp_kwargs({"trainer_param": {"max_epochs": 3}})


def test_mlp_kwargs_known_sections_pass():
    """Every section the suite reads validates, and empty / None kwargs pass."""
    validate_nested_mlp_kwargs(
        {
            "model_params": {"learning_rate": 1e-3},
            "network_params": {"nlayers": 2},
            "trainer_params": {"max_epochs": 3},
            "dataloader_params": {"batch_size": 64},
            "datamodule_params": {},
            "use_swa": True,
            "swa_params": {"swa_lrs": 1e-4},
            "tune_params": False,
            "float32_matmul_precision": "HIGH",
        }
    )
    validate_nested_mlp_kwargs({})
    validate_nested_mlp_kwargs(None)


def test_mlp_matmul_precision_is_validated_and_normalised():
    """An unsupported precision raises; a supported one is lower-cased; absent stays None."""
    with pytest.raises(ValidationError):
        MLPConfig(float32_matmul_precision="ultra")
    assert MLPConfig(float32_matmul_precision="HIGH").float32_matmul_precision == "high"
    assert MLPConfig().float32_matmul_precision is None


def test_mlp_defaults_are_what_the_suite_does_without_the_key():
    """Absent keys mean: SWA off and no tuning, exactly the suite's own fallbacks."""
    cfg = MLPConfig()
    assert cfg.use_swa is False and cfg.tune_params is False


def test_get_training_configs_rejects_an_unknown_mlp_section():
    """The suite's config factory applies the check, so the error surfaces before any model is built."""
    from mlframe.training._helpers_training_configs import get_training_configs

    with pytest.raises(ValidationError, match="dataloader_param"):
        get_training_configs(mlp_kwargs={"dataloader_param": {"batch_size": 8}})


# ---- LinearModelConfig -----------------------------------------------------


def test_linear_alpha_negative_raises():
    """Linear alpha negative raises."""
    with pytest.raises(ValidationError):
        LinearModelConfig(alpha=-1.0)


def test_linear_l1_ratio_above_one_raises():
    """Linear l1 ratio above one raises."""
    with pytest.raises(ValidationError):
        LinearModelConfig(model_type="elasticnet", l1_ratio=1.5)


def test_linear_l1_ratio_negative_raises():
    """Linear l1 ratio negative raises."""
    with pytest.raises(ValidationError):
        LinearModelConfig(l1_ratio=-0.2)


def test_linear_alpha_zero_is_ols_and_valid():
    """Linear alpha zero is ols and valid."""
    assert LinearModelConfig(alpha=0.0).alpha == 0.0


# ---- MultilabelDispatchConfig ----------------------------------------------


def test_multilabel_n_chains_zero_raises():
    """Multilabel n chains zero raises."""
    with pytest.raises(ValidationError):
        MultilabelDispatchConfig(n_chains=0)


def test_multilabel_cv_one_raises():
    """Multilabel cv one raises."""
    with pytest.raises(ValidationError):
        MultilabelDispatchConfig(cv=1)


def test_multilabel_cv_none_is_valid():
    """Multilabel cv none is valid."""
    assert MultilabelDispatchConfig(cv=None).cv is None
