"""The ``ModelHyperparamsConfig`` knobs forwarded to ``get_training_configs`` are declared fields that match its parameters.

They were once undeclared pass-through extras; they are declared now, as ``Optional[...] = None`` ("not set": the suite drops None-valued fields, so
``get_training_configs`` applies its own default). A field with no matching parameter would be forwarded and crash, and a value outside the
parameter's own domain would only fail deep inside a fit.
"""

from __future__ import annotations

import inspect

import pytest

from mlframe.training._helpers_training_configs import get_training_configs
from mlframe.training.configs import ModelHyperparamsConfig

FORWARDED = (
    "method",
    "mae_weight",
    "std_weight",
    "roc_auc_weight",
    "pr_auc_weight",
    "brier_loss_weight",
    "min_roc_auc",
    "roc_auc_penalty",
    "use_weighted_calibration",
    "weight_by_class_npositives",
    "nbins",
    "robustness_num_ts_splits",
    "robustness_std_coeff",
    "robustness_greater_is_better",
    "validation_fraction",
    "use_explicit_early_stopping",
    "random_seed",
    "verbose",
    "catboost_custom_regr_metrics",
)


@pytest.mark.parametrize("name", FORWARDED)
def test_the_field_is_a_parameter_of_get_training_configs_and_unset_by_default(name):
    """Declared on the config, accepted by ``get_training_configs``, and ``None`` until the caller sets it."""
    assert name in ModelHyperparamsConfig.model_fields
    assert name in inspect.signature(get_training_configs).parameters
    assert ModelHyperparamsConfig.model_fields[name].default is None


def test_a_default_config_forwards_none_of_them():
    """Nothing is forwarded unless set, so ``get_training_configs`` keeps its defaults."""
    dumped = ModelHyperparamsConfig().model_dump(exclude_none=True)
    assert not (set(FORWARDED) & set(dumped))


def test_a_set_knob_reaches_get_training_configs():
    """The value the caller sets is what the factory receives."""
    cfg = ModelHyperparamsConfig(mae_weight=7.5, nbins=20)
    forwarded = {k: v for k, v in cfg.model_dump(exclude_none=True).items() if k in FORWARDED}
    assert forwarded == {"mae_weight": 7.5, "nbins": 20}
    assert get_training_configs(**forwarded) is not None
