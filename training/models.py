"""
Model configuration and linear model implementations for mlframe.

This module contains:
- Linear model wrappers (linear, ridge, lasso, elasticnet, huber, ransac, sgd)
- Model factory functions
- Model type detection utilities
"""

import logging
from typing import Callable, Dict

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    Lasso,
    ElasticNet,
    HuberRegressor,
    RANSACRegressor,
    SGDRegressor,
    SGDClassifier,
)
from sklearn.calibration import CalibratedClassifierCV

from .configs import LinearModelConfig

logger = logging.getLogger(__name__)


# ==================================================================================
# CONSTANTS
# ==================================================================================

LINEAR_MODEL_TYPES = {"linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"}
TREE_MODEL_TYPES = {"cb", "lgb", "xgb", "hgb", "rf", "et", "gb"}
NEURAL_MODEL_TYPES = {"mlp", "nn"}
VALID_SGD_CLASSIFICATION_LOSSES = {"hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"}


# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================


def _get_sklearn_verbose(verbose: int) -> int:
    """Convert mlframe verbose level to sklearn verbose level."""
    return verbose - 1 if verbose > 0 else 0


def _get_l1_ratio(config: LinearModelConfig) -> float:
    """Get l1_ratio for elasticnet penalty, else default."""
    return config.l1_ratio if config.penalty == "elasticnet" else 0.15


# ==================================================================================
# REGRESSION MODEL BUILDERS
# ==================================================================================


def _build_linear_regressor(config: LinearModelConfig) -> BaseEstimator:
    return LinearRegression(n_jobs=config.n_jobs)


def _build_ridge_regressor(config: LinearModelConfig) -> BaseEstimator:
    return Ridge(
        alpha=config.alpha,
        random_state=config.random_state,
        max_iter=config.max_iter,
    )


def _build_lasso_regressor(config: LinearModelConfig) -> BaseEstimator:
    return Lasso(
        alpha=config.alpha,
        random_state=config.random_state,
        max_iter=config.max_iter,
        tol=config.tol,
    )


def _build_elasticnet_regressor(config: LinearModelConfig) -> BaseEstimator:
    return ElasticNet(
        alpha=config.alpha,
        l1_ratio=config.l1_ratio,
        random_state=config.random_state,
        max_iter=config.max_iter,
        tol=config.tol,
    )


def _build_huber_regressor(config: LinearModelConfig) -> BaseEstimator:
    return HuberRegressor(
        epsilon=config.epsilon,
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
    )


def _build_ransac_regressor(config: LinearModelConfig) -> BaseEstimator:
    return RANSACRegressor(
        estimator=LinearRegression(),
        max_trials=config.max_trials,
        residual_threshold=config.residual_threshold,
        random_state=config.random_state,
    )


def _build_sgd_regressor(config: LinearModelConfig) -> BaseEstimator:
    return SGDRegressor(
        loss=config.loss,
        penalty=config.penalty,
        alpha=config.alpha,
        l1_ratio=_get_l1_ratio(config),
        max_iter=config.max_iter,
        tol=config.tol,
        learning_rate=config.learning_rate,
        eta0=config.eta0,
        random_state=config.random_state,
        verbose=_get_sklearn_verbose(config.verbose),
    )


_REGRESSION_BUILDERS: Dict[str, Callable[[LinearModelConfig], BaseEstimator]] = {
    "linear": _build_linear_regressor,
    "ridge": _build_ridge_regressor,
    "lasso": _build_lasso_regressor,
    "elasticnet": _build_elasticnet_regressor,
    "huber": _build_huber_regressor,
    "ransac": _build_ransac_regressor,
    "sgd": _build_sgd_regressor,
}


# ==================================================================================
# CLASSIFICATION MODEL BUILDERS
# ==================================================================================


def _build_linear_classifier(config: LinearModelConfig) -> BaseEstimator:
    return LogisticRegression(
        C=config.C,
        solver=config.solver,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=_get_sklearn_verbose(config.verbose),
    )


def _build_ridge_classifier(config: LinearModelConfig) -> BaseEstimator:
    return RidgeClassifier(
        alpha=config.alpha,
        random_state=config.random_state,
        max_iter=config.max_iter,
        tol=config.tol,
    )


def _build_lasso_classifier(config: LinearModelConfig) -> BaseEstimator:
    """Lasso classification via LogisticRegression with L1 penalty."""
    return LogisticRegression(
        penalty="l1",
        C=1.0 / config.alpha if config.alpha > 0 else 1.0,
        solver="saga",
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=_get_sklearn_verbose(config.verbose),
    )


def _build_elasticnet_classifier(config: LinearModelConfig) -> BaseEstimator:
    """ElasticNet classification via LogisticRegression with elasticnet penalty."""
    return LogisticRegression(
        penalty="elasticnet",
        C=1.0 / config.alpha if config.alpha > 0 else 1.0,
        l1_ratio=config.l1_ratio,
        solver="saga",
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=_get_sklearn_verbose(config.verbose),
    )


def _build_huber_classifier(config: LinearModelConfig) -> BaseEstimator:
    """Huber loss classification via SGDClassifier with modified_huber loss."""
    return SGDClassifier(
        loss="modified_huber",
        penalty=config.penalty,
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        learning_rate=config.learning_rate,
        eta0=config.eta0,
        random_state=config.random_state,
        verbose=_get_sklearn_verbose(config.verbose),
    )


def _build_ransac_classifier(config: LinearModelConfig) -> BaseEstimator:
    """RANSAC for classification falls back to LogisticRegression."""
    logger.warning("RANSAC is primarily for regression. Using LogisticRegression for classification.")
    return LogisticRegression(
        max_iter=config.max_iter,
        random_state=config.random_state,
    )


def _build_sgd_classifier(config: LinearModelConfig) -> BaseEstimator:
    loss = config.loss if config.loss in VALID_SGD_CLASSIFICATION_LOSSES else "log_loss"
    return SGDClassifier(
        loss=loss,
        penalty=config.penalty,
        alpha=config.alpha,
        l1_ratio=_get_l1_ratio(config),
        max_iter=config.max_iter,
        tol=config.tol,
        learning_rate=config.learning_rate,
        eta0=config.eta0,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=_get_sklearn_verbose(config.verbose),
    )


_CLASSIFICATION_BUILDERS: Dict[str, Callable[[LinearModelConfig], BaseEstimator]] = {
    "linear": _build_linear_classifier,
    "ridge": _build_ridge_classifier,
    "lasso": _build_lasso_classifier,
    "elasticnet": _build_elasticnet_classifier,
    "huber": _build_huber_classifier,
    "ransac": _build_ransac_classifier,
    "sgd": _build_sgd_classifier,
}


# ==================================================================================
# LINEAR MODEL FACTORY
# ==================================================================================


def create_linear_model(
    model_type: str,
    config: LinearModelConfig,
    use_regression: bool = True,
) -> BaseEstimator:
    """
    Create a linear model based on configuration.

    Args:
        model_type: Model type codename (linear, ridge, lasso, elasticnet, huber, ransac, sgd)
        config: Linear model configuration
        use_regression: Whether to use regression (True) or classification (False)

    Returns:
        Configured sklearn estimator
    """
    model_type = model_type.lower()
    builders = _REGRESSION_BUILDERS if use_regression else _CLASSIFICATION_BUILDERS

    if model_type not in builders:
        task = "regression" if use_regression else "classification"
        raise ValueError(f"Unknown {task} model type: {model_type}")

    model = builders[model_type](config)

    # Apply calibration for classifiers if requested
    if not use_regression and config.use_calibrated_classifier and hasattr(model, "predict_proba"):
        if config.verbose:
            logger.info(f"Wrapping {model_type} with CalibratedClassifierCV")
        model = CalibratedClassifierCV(model, cv=3, method="isotonic")

    return model


# ==================================================================================
# MODEL TYPE DETECTION
# ==================================================================================


def is_linear_model(model_name: str) -> bool:
    """Check if a model name corresponds to a linear model."""
    return model_name.lower() in LINEAR_MODEL_TYPES


def is_tree_model(model_name: str) -> bool:
    """Check if a model name corresponds to a tree-based model."""
    return model_name.lower() in TREE_MODEL_TYPES


def is_neural_model(model_name: str) -> bool:
    """Check if a model name corresponds to a neural network model."""
    return model_name.lower() in NEURAL_MODEL_TYPES


# ==================================================================================
# EXPORTS
# ==================================================================================

__all__ = [
    # Model creation
    "create_linear_model",
    # Model type detection
    "is_linear_model",
    "is_tree_model",
    "is_neural_model",
    # Constants
    "LINEAR_MODEL_TYPES",
    "TREE_MODEL_TYPES",
    "NEURAL_MODEL_TYPES",
]
