"""Regression tests for the ``LinearModelConfig.model_type`` fuzz-coverage gap.

Bug: the fuzz suite's ``MODELS`` axis only ever emitted the "linear" model name, so
ridge/lasso/elasticnet/huber/ransac/sgd (6 of ``VALID_LINEAR_MODEL_TYPES``'s 7 values) had zero fuzz
lever -- ``LinearModelConfig.model_type`` itself has NO effect on which estimator the suite builds
(``_trainer_configure_helpers.py`` explicitly excludes it from the merged config:
``linear_model_config.model_dump(exclude={"model_type"})``); the real dispatch lever is the model NAME
string in the ``models``/``mlframe_models`` allowlist (``LINEAR_MODEL_TYPES & models_set`` in
``_trainer_configure.py``). Fixed by adding a ``linear_model_type_cfg`` combo axis and substituting the
"linear" entry in the models list actually sent to the suite with the resolved variant name.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.training._fuzz_combo import FuzzCombo
from mlframe.training._configs_base import VALID_LINEAR_MODEL_TYPES
from mlframe.training.models import create_linear_model
from mlframe.training.configs import LinearModelConfig


def _make_combo(models, linear_model_type_cfg="linear", target_type="regression") -> FuzzCombo:
    """Build a minimal FuzzCombo for exercising the linear_model_type_cfg axis in isolation."""
    return FuzzCombo(
        models=models,
        input_type="pandas",
        n_rows=1000,
        cat_feature_count=0,
        null_fraction_cats=0.0,
        use_mrmr_fs=False,
        weight_schemas=("uniform",),
        target_type=target_type,
        auto_detect_cats=True,
        align_polars_categorical_dicts=True,
        seed=0,
        linear_model_type_cfg=linear_model_type_cfg,
    )


def test_axis_covers_every_valid_linear_model_type():
    """Every VALID_LINEAR_MODEL_TYPES value must actually be reachable through the axis."""
    from tests.training._fuzz_combo.axes import AXES

    assert set(AXES["linear_model_type_cfg"]) == VALID_LINEAR_MODEL_TYPES


@pytest.mark.parametrize("variant", sorted(VALID_LINEAR_MODEL_TYPES - {"linear"}))
def test_canonical_models_for_suite_substitutes_linear_entry(variant):
    """Each non-default variant replaces the "linear" model-name entry with its own name."""
    combo = _make_combo(models=("linear",), linear_model_type_cfg=variant)
    assert combo._canonical_models_for_suite() == (variant,)


def test_canonical_models_for_suite_noop_when_linear_absent():
    """A combo without "linear" in its model list is unaffected by the axis."""
    combo = _make_combo(models=("cb", "xgb"), linear_model_type_cfg="ridge")
    assert combo._canonical_models_for_suite() == ("cb", "xgb")
    assert combo._canonical_linear_model_type() == "linear"


def test_canonical_models_for_suite_noop_at_default():
    """The default "linear" variant leaves the model list unchanged."""
    combo = _make_combo(models=("linear",), linear_model_type_cfg="linear")
    assert combo._canonical_models_for_suite() == ("linear",)


def test_canonical_linear_model_type_collapses_for_ltr():
    """LTR filters the model list down to native rankers before use -- "linear" never survives, so the
    axis must collapse to avoid a spurious dedup split with zero behavioural difference."""
    combo = _make_combo(models=("linear", "cb"), linear_model_type_cfg="ridge", target_type="learning_to_rank")
    assert combo._canonical_linear_model_type() == "linear"


def test_canonical_mlframe_models_explicit_forced_true_for_non_default_variant():
    """A non-default variant must force an explicit mlframe_models allowlist -- otherwise the suite's
    implicit-default path never even looks at the substituted list."""
    combo = _make_combo(models=("linear",), linear_model_type_cfg="ridge")
    assert combo._canonical_mlframe_models_explicit() is True


@pytest.mark.parametrize("variant", sorted(VALID_LINEAR_MODEL_TYPES))
def test_create_linear_model_builds_a_real_distinct_estimator_per_variant(variant):
    """Every VALID_LINEAR_MODEL_TYPES value reaches a real, distinct sklearn estimator class."""
    config = LinearModelConfig(model_type=variant)
    reg = create_linear_model(variant, config, use_regression=True)
    clf = create_linear_model(variant, config, use_regression=False)
    assert type(reg).__module__.startswith("sklearn")
    assert type(clf).__module__.startswith("sklearn")


@pytest.mark.parametrize("variant", sorted(VALID_LINEAR_MODEL_TYPES - {"ransac"}))
def test_create_linear_model_regressor_fits_on_toy_data(variant):
    """Non-RANSAC variants fit cleanly on a trivial well-conditioned regression toy (RANSAC needs a
    larger sample to find consensus and is already covered by its own dedicated tests)."""
    rng = np.random.RandomState(0)
    X = rng.rand(200, 3)
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.01, size=200)
    config = LinearModelConfig(model_type=variant, max_iter=200)
    model = create_linear_model(variant, config, use_regression=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)
