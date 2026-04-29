"""C2 — meta-test that every value in ``VALID_LINEAR_MODEL_TYPES`` and
every linear-model alias accepted via the public ``create_linear_model``
factory ACTUALLY trains on a tiny synthetic dataset without raising.

The structural test ``test_strategy_registration.py`` (MT-1) already
gates "name in registry → strategy class exists". This test goes one
step further: "name accepted by validator → fit + predict actually
work end-to-end on minimal data". Catches the failure mode where a
strategy class exists but its inner constructor blows up because (e.g.)
a kwarg the strategy passes was renamed in a sklearn upgrade.

Tree models (cb / lgb / xgb / hgb / mlp / ngb) are NOT covered here —
they have heavyweight dependencies (CatBoost / LGBM / XGBoost C++
backends) and per-strategy quirks that make a generic "train one of
each" expensive enough to blow our 1-min meta-test budget. They're
covered by ``tests/training/test_core.py`` (full integration suite).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.configs import (
    VALID_LINEAR_MODEL_TYPES,
    LinearModelConfig,
)
from mlframe.training.models import create_linear_model


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(123)
    n_rows, n_features = 100, 5
    X = rng.standard_normal((n_rows, n_features))
    y = X @ rng.standard_normal(n_features) + 0.1 * rng.standard_normal(n_rows)
    return X, y


@pytest.fixture(scope="module")
def binary_data():
    rng = np.random.default_rng(123)
    n_rows, n_features = 100, 5
    X = rng.standard_normal((n_rows, n_features))
    y = (X @ rng.standard_normal(n_features) > 0).astype(np.int64)
    return X, y


# Linear-model values from the validator that the regression-mode factory
# can actually build. ``ransac`` and ``huber`` are robust regressors;
# ``sgd`` works in both regression and classification with appropriate
# loss.
_REGRESSION_TYPES = sorted(VALID_LINEAR_MODEL_TYPES)


@pytest.mark.parametrize("model_type", _REGRESSION_TYPES)
def test_every_linear_regression_type_fits_and_predicts(model_type, regression_data):
    """Every member of ``VALID_LINEAR_MODEL_TYPES`` builds, fits on a
    tiny dataset, and emits predictions of the expected shape.
    """
    X, y = regression_data
    cfg_kwargs = {"model_type": model_type, "random_state": 42, "max_iter": 200}
    # ``ransac`` / ``huber`` need extra knobs to converge on short data.
    if model_type == "ransac":
        cfg_kwargs["max_trials"] = 50
    cfg = LinearModelConfig(**cfg_kwargs)
    model = create_linear_model(model_type, cfg, use_regression=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],), (
        f"{model_type}: predictions shape {preds.shape} != {(X.shape[0],)}"
    )
    assert np.isfinite(preds).all(), (
        f"{model_type}: predictions contain non-finite values"
    )


# Classification: only the types that have a classifier-side builder.
# ``ransac`` / ``huber`` are regression-only; ``elasticnet`` / ``lasso``
# don't have native classifiers in sklearn (only ``ridge`` /
# ``logistic`` / ``sgd`` are practical).
_CLASSIFICATION_TYPES = ("ridge", "sgd")


@pytest.mark.parametrize("model_type", _CLASSIFICATION_TYPES)
def test_every_linear_classification_type_fits_and_predicts(model_type, binary_data):
    X, y = binary_data
    cfg_kwargs = {"model_type": model_type, "random_state": 42, "max_iter": 200}
    if model_type == "sgd":
        cfg_kwargs["loss"] = "log_loss"
    cfg = LinearModelConfig(**cfg_kwargs)
    model = create_linear_model(model_type, cfg, use_regression=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],), (
        f"{model_type}: predictions shape {preds.shape} != {(X.shape[0],)}"
    )
    assert set(np.unique(preds)) <= {0, 1}, (
        f"{model_type}: classification predictions not in {{0, 1}}"
    )


def test_valid_linear_model_types_set_is_non_empty():
    """Sanity: the validator allow-list isn't accidentally empty."""
    assert VALID_LINEAR_MODEL_TYPES, "VALID_LINEAR_MODEL_TYPES is empty"
    assert "ridge" in VALID_LINEAR_MODEL_TYPES, (
        "VALID_LINEAR_MODEL_TYPES dropped 'ridge' — that's the canonical default"
    )
