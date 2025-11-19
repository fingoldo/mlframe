"""
Model configuration, training, and linear model implementations for mlframe.

This module contains:
- Linear model wrappers (linear, ridge, lasso, elasticnet, huber, ransac, sgd)
- Model training functions
- Configuration utilities
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Any, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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

    if use_regression:
        # ============ REGRESSION MODELS ============
        if model_type == "linear":
            model = LinearRegression(
                n_jobs=config.n_jobs,
            )

        elif model_type == "ridge":
            model = Ridge(
                alpha=config.alpha,
                random_state=config.random_state,
                max_iter=config.max_iter,
            )

        elif model_type == "lasso":
            model = Lasso(
                alpha=config.alpha,
                random_state=config.random_state,
                max_iter=config.max_iter,
                tol=config.tol,
            )

        elif model_type == "elasticnet":
            model = ElasticNet(
                alpha=config.alpha,
                l1_ratio=config.l1_ratio,
                random_state=config.random_state,
                max_iter=config.max_iter,
                tol=config.tol,
            )

        elif model_type == "huber":
            model = HuberRegressor(
                epsilon=config.epsilon,
                alpha=config.alpha,
                max_iter=config.max_iter,
                tol=config.tol,
            )

        elif model_type == "ransac":
            base_estimator = LinearRegression()
            model = RANSACRegressor(
                estimator=base_estimator,
                max_trials=config.max_trials,
                residual_threshold=config.residual_threshold,
                random_state=config.random_state,
            )

        elif model_type == "sgd":
            model = SGDRegressor(
                loss=config.loss,
                penalty=config.penalty,
                alpha=config.alpha,
                l1_ratio=config.l1_ratio if config.penalty == 'elasticnet' else 0.15,
                max_iter=config.max_iter,
                tol=config.tol,
                learning_rate=config.learning_rate,
                eta0=config.eta0,
                random_state=config.random_state,
                verbose=config.verbose - 1 if config.verbose > 0 else 0,
            )

        else:
            raise ValueError(f"Unknown regression model type: {model_type}")

    else:
        # ============ CLASSIFICATION MODELS ============
        if model_type == "linear":
            model = LogisticRegression(
                C=config.C,
                solver=config.solver,
                max_iter=config.max_iter,
                tol=config.tol,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose - 1 if config.verbose > 0 else 0,
            )

        elif model_type == "ridge":
            model = RidgeClassifier(
                alpha=config.alpha,
                random_state=config.random_state,
                max_iter=config.max_iter,
                tol=config.tol,
            )

        elif model_type == "lasso" or model_type == "elasticnet":
            # Lasso and ElasticNet for classification use LogisticRegression with penalty
            penalty = 'l1' if model_type == "lasso" else 'elasticnet'
            solver = 'saga'  # saga supports both l1 and elasticnet

            model = LogisticRegression(
                penalty=penalty,
                C=1.0 / config.alpha if config.alpha > 0 else 1.0,
                l1_ratio=config.l1_ratio if model_type == "elasticnet" else None,
                solver=solver,
                max_iter=config.max_iter,
                tol=config.tol,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose - 1 if config.verbose > 0 else 0,
            )

        elif model_type == "huber":
            # Huber loss for classification - use SGD with modified_huber loss
            model = SGDClassifier(
                loss='modified_huber',
                penalty=config.penalty,
                alpha=config.alpha,
                max_iter=config.max_iter,
                tol=config.tol,
                learning_rate=config.learning_rate,
                eta0=config.eta0,
                random_state=config.random_state,
                verbose=config.verbose - 1 if config.verbose > 0 else 0,
            )

        elif model_type == "ransac":
            # RANSAC for classification - wrap base classifier
            base_estimator = LogisticRegression(max_iter=config.max_iter, random_state=config.random_state)
            # RANSAC doesn't directly support classification, use it as an outlier detector
            # and fall back to regular logistic regression
            logger.warning(f"RANSAC is primarily for regression. Using LogisticRegression for classification.")
            model = base_estimator

        elif model_type == "sgd":
            model = SGDClassifier(
                loss=config.loss if config.loss in ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'] else 'log_loss',
                penalty=config.penalty,
                alpha=config.alpha,
                l1_ratio=config.l1_ratio if config.penalty == 'elasticnet' else 0.15,
                max_iter=config.max_iter,
                tol=config.tol,
                learning_rate=config.learning_rate,
                eta0=config.eta0,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose - 1 if config.verbose > 0 else 0,
            )

        else:
            raise ValueError(f"Unknown classification model type: {model_type}")

        # Apply calibration if requested
        if config.use_calibrated_classifier and hasattr(model, 'predict_proba'):
            if config.verbose:
                logger.info(f"Wrapping {model_type} with CalibratedClassifierCV")
            model = CalibratedClassifierCV(model, cv=3, method='sigmoid')

    return model


# ==================================================================================
# MODEL TYPE DETECTION
# ==================================================================================


LINEAR_MODEL_TYPES = {
    'linear',
    'ridge',
    'lasso',
    'elasticnet',
    'huber',
    'ransac',
    'sgd',
}


def is_linear_model(model_name: str) -> bool:
    """Check if a model name corresponds to a linear model."""
    return model_name.lower() in LINEAR_MODEL_TYPES


def is_tree_model(model_name: str) -> bool:
    """Check if a model name corresponds to a tree-based model."""
    tree_models = {'cb', 'lgb', 'xgb', 'hgb', 'rf', 'et', 'gb'}
    return model_name.lower() in tree_models


def is_neural_model(model_name: str) -> bool:
    """Check if a model name corresponds to a neural network model."""
    neural_models = {'mlp', 'nn'}
    return model_name.lower() in neural_models


# ==================================================================================
# MODEL TRAINING (RE-EXPORT FROM MAIN MODULE FOR NOW)
# ==================================================================================

try:
    from ..training import (
        train_and_evaluate_model,
        process_model,
        configure_training_params,
        get_training_configs,
        select_target,
    )

    _IMPORTED_TRAINING_FUNCS = True
except ImportError:
    _IMPORTED_TRAINING_FUNCS = False
    logger.warning("Could not import training functions from main module")


# ==================================================================================
# SIMPLIFIED TRAINING FUNCTION FOR LINEAR MODELS
# ==================================================================================


def train_linear_model(
    model_type: str,
    train_df: Union[pd.DataFrame, np.ndarray],
    train_target: Union[pd.Series, np.ndarray],
    val_df: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    val_target: Optional[Union[pd.Series, np.ndarray]] = None,
    test_df: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    test_target: Optional[Union[pd.Series, np.ndarray]] = None,
    config: Optional[LinearModelConfig] = None,
    use_regression: bool = True,
    metamodel_func: Optional[Callable] = None,
    verbose: int = 1,
) -> dict:
    """
    Train a linear model with optional target transformation.

    Args:
        model_type: Linear model type (linear, ridge, lasso, etc.)
        train_df: Training features
        train_target: Training target
        val_df: Validation features (optional)
        val_target: Validation target (optional)
        test_df: Test features (optional)
        test_target: Test target (optional)
        config: Linear model configuration
        use_regression: Whether to use regression
        metamodel_func: Function to wrap model for target transformation
        verbose: Verbosity level

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if config is None:
        config = LinearModelConfig(model_type=model_type)

    # Create the model
    model = create_linear_model(model_type, config, use_regression=use_regression)

    # Apply metamodel transformation if provided
    if metamodel_func is not None:
        model = metamodel_func(model)

    # Train the model
    if verbose:
        logger.info(f"Training {model_type} model...")

    model.fit(train_df, train_target)

    # Generate predictions
    result = {
        'model': model,
        'model_type': model_type,
        'train_preds': model.predict(train_df),
    }

    if hasattr(model, 'predict_proba') and not use_regression:
        result['train_probs'] = model.predict_proba(train_df)

    if val_df is not None and val_target is not None:
        result['val_preds'] = model.predict(val_df)
        if hasattr(model, 'predict_proba') and not use_regression:
            result['val_probs'] = model.predict_proba(val_df)

    if test_df is not None and test_target is not None:
        result['test_preds'] = model.predict(test_df)
        if hasattr(model, 'predict_proba') and not use_regression:
            result['test_probs'] = model.predict_proba(test_df)

    if verbose:
        logger.info(f"{model_type} model training completed")

    return result


# ==================================================================================
# EXPORTS
# ==================================================================================

__all__ = [
    # Model creation
    'create_linear_model',
    'train_linear_model',

    # Model type detection
    'is_linear_model',
    'is_tree_model',
    'is_neural_model',
    'LINEAR_MODEL_TYPES',
]

# Conditional exports
if _IMPORTED_TRAINING_FUNCS:
    __all__.extend([
        'train_and_evaluate_model',
        'process_model',
        'configure_training_params',
        'get_training_configs',
        'select_target',
    ])
