"""Unit tests for `training.models` public API.

Covers the three model-type predicates (`is_linear_model`, `is_tree_model`,
`is_neural_model`) and the `create_linear_model` factory. The model-type
constants (LINEAR_MODEL_TYPES / TREE_MODEL_TYPES / NEURAL_MODEL_TYPES) are
the contract -- predicates pin which names route to which family for the
suite-level dispatch.
"""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)

from mlframe.training.configs import LinearModelConfig
from mlframe.training.models import (
    LINEAR_MODEL_TYPES,
    NEURAL_MODEL_TYPES,
    TREE_MODEL_TYPES,
    create_linear_model,
    is_linear_model,
    is_neural_model,
    is_tree_model,
)


@pytest.mark.parametrize("name", sorted(LINEAR_MODEL_TYPES))
def test_is_linear_model_true_for_documented_names(name):
    assert is_linear_model(name) is True


@pytest.mark.parametrize("name", sorted(TREE_MODEL_TYPES))
def test_is_tree_model_true_for_documented_names(name):
    assert is_tree_model(name) is True


@pytest.mark.parametrize("name", sorted(NEURAL_MODEL_TYPES))
def test_is_neural_model_true_for_documented_names(name):
    assert is_neural_model(name) is True


@pytest.mark.parametrize(
    "name,linear,tree,neural",
    [
        ("ridge", True, False, False),
        ("lasso", True, False, False),
        ("cb", False, True, False),
        ("lgb", False, True, False),
        ("xgb", False, True, False),
        ("rf", False, True, False),
        ("mlp", False, False, True),
        ("lstm", False, False, True),
        ("transformer", False, False, True),
    ],
)
def test_predicates_route_canonical_names_to_correct_family(name, linear, tree, neural):
    assert is_linear_model(name) is linear
    assert is_tree_model(name) is tree
    assert is_neural_model(name) is neural


@pytest.mark.parametrize("name", ["unknown", "random", "totally_made_up", ""])
def test_predicates_return_false_for_unknown_names(name):
    # Unknown names must NOT raise; the predicates are dispatch gates and
    # callers rely on a clean False to fall through to other branches.
    assert is_linear_model(name) is False
    assert is_tree_model(name) is False
    assert is_neural_model(name) is False


@pytest.mark.parametrize("name", ["Ridge", "RIDGE", "RiDgE", "CB", "Cb", "MLP", "Mlp"])
def test_predicates_are_case_insensitive(name):
    # The docstrings explicitly document case-insensitivity ("model_type
    # is case-insensitive and normalized to lowercase"). Any of these
    # must produce the same result as the lowercase canonical name.
    assert is_linear_model(name) == is_linear_model(name.lower())
    assert is_tree_model(name) == is_tree_model(name.lower())
    assert is_neural_model(name) == is_neural_model(name.lower())


def test_predicate_sets_are_disjoint():
    # The three model families partition the canonical name space; a
    # name silently appearing in two sets would cause the dispatch to
    # collide. Pin the invariant explicitly.
    assert not (LINEAR_MODEL_TYPES & TREE_MODEL_TYPES)
    assert not (LINEAR_MODEL_TYPES & NEURAL_MODEL_TYPES)
    assert not (TREE_MODEL_TYPES & NEURAL_MODEL_TYPES)


# ----- create_linear_model factory ----------------------------------------


@pytest.mark.parametrize(
    "model_type,expected_regressor_cls",
    [
        ("linear", Ridge),  # see _LINEAR_DEFAULT_RIDGE_ALPHA -- "linear" routes to Ridge(1e-3)
        ("ridge", Ridge),
        ("lasso", Lasso),
        ("elasticnet", ElasticNet),
        ("huber", HuberRegressor),
        ("ransac", RANSACRegressor),
        ("sgd", SGDRegressor),
    ],
)
def test_create_linear_model_regression_returns_correct_estimator(model_type, expected_regressor_cls):
    config = LinearModelConfig(alpha=0.1, max_iter=500)
    model = create_linear_model(model_type, config, use_regression=True)
    assert isinstance(model, expected_regressor_cls)
    # sklearn API contract.
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
    assert isinstance(model, BaseEstimator)


def test_create_linear_model_unknown_type_raises():
    config = LinearModelConfig()
    with pytest.raises(ValueError, match="Unknown regression model type"):
        create_linear_model("not_a_model", config, use_regression=True)


def test_create_linear_model_unknown_type_for_classification_raises():
    config = LinearModelConfig()
    with pytest.raises(ValueError, match="Unknown classification model type"):
        create_linear_model("totally_bogus", config, use_regression=False)


def test_create_linear_model_classification_returns_classifier():
    config = LinearModelConfig(alpha=1.0, max_iter=300, use_calibrated_classifier=False)
    model = create_linear_model("sgd", config, use_regression=False)
    assert hasattr(model, "fit")
    # SGD classification routes to SGDClassifier.
    assert isinstance(model, SGDClassifier)


def test_create_linear_model_is_case_insensitive():
    config = LinearModelConfig()
    upper = create_linear_model("RIDGE", config, use_regression=True)
    lower = create_linear_model("ridge", config, use_regression=True)
    assert type(upper) is type(lower)


def test_create_linear_model_fits_on_trivial_data():
    # Smoke test: the returned estimator must actually fit + predict on a
    # tiny synthetic without raising. Validates the config knobs make it
    # through to the underlying sklearn class.
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=50)

    config = LinearModelConfig(alpha=0.1, max_iter=500)
    model = create_linear_model("ridge", config, use_regression=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (50,)
    # Ridge on well-conditioned data should recover the linear signal
    # closely; pin a generous floor (R^2 >= 0.9).
    from sklearn.metrics import r2_score

    assert r2_score(y, preds) >= 0.9
