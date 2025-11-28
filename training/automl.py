"""
AutoML model training for mlframe.

Separate module for AutoGluon and LightAutoML (LAMA) training.
These models require the target to be in the dataframe, so they work differently
from the regular mlframe pipeline.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd, polars as pl
from typing import Optional, Dict, Any, Union
from types import SimpleNamespace
from pyutilz.system import clean_ram
from sklearn.metrics import roc_auc_score

from .configs import AutoMLConfig
from .utils import log_ram_usage, get_pandas_view_of_polars_df

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

LOG_SEPARATOR = "=" * 80


def train_autogluon_model(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_name: str = "target",
    init_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
) -> Optional[SimpleNamespace]:
    """
    Train an AutoGluon model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame (must include target column).
    test_df : pd.DataFrame, optional
        Test DataFrame for evaluation (must include target column).
    target_name : str, default="target"
        Name of the target column.
    init_params : dict, optional
        Parameters for TabularPredictor initialization.
    fit_params : dict, optional
        Parameters for the fit() method.
    verbose : int, default=1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Returns
    -------
    SimpleNamespace or None
        Returns None if AutoGluon is not installed.
        Otherwise returns SimpleNamespace with:
        - model: Trained TabularPredictor
        - metrics: dict with evaluation metrics (e.g., test_auc)
        - fi: Feature importance DataFrame
        - test_target: Test target values (if test_df provided)
        - test_probs: Prediction probabilities (if test_df provided)

    Raises
    ------
    ImportError
        If AutoGluon is not installed (returns None instead of raising).
    """
    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
    except ImportError as e:
        logger.error(f"AutoGluon not available: {e}")
        return None

    init_params = init_params or {}
    fit_params = fit_params or {}

    if verbose:
        logger.info(f"Training AutoGluon model on {len(train_df)} rows...")

    # Create predictor
    predictor = TabularPredictor(label=target_name, verbosity=verbose, **init_params)

    # Fit model
    train_data = TabularDataset(train_df)
    predictor.fit(train_data, **fit_params)

    clean_ram()
    if verbose:
        log_ram_usage()

    # Evaluate on test set if provided
    test_probs = None
    test_target = None
    metrics = {}

    if test_df is not None and len(test_df) > 0:
        if target_name in test_df.columns:
            test_target = test_df[target_name]
            test_features = test_df.drop(columns=[target_name])
        else:
            test_features = test_df

        test_probs = predictor.predict_proba(test_features)
        if isinstance(test_probs, pd.DataFrame):
            test_probs = test_probs.to_numpy()

        if test_target is not None and verbose:
            try:
                auc = roc_auc_score(test_target, test_probs[:, 1])
                logger.info(f"AutoGluon test AUC: {auc:.4f}")
                metrics["test_auc"] = auc
            except (ValueError, TypeError) as e:
                # ValueError: Only one class present in y_true or invalid input
                # TypeError: Cannot compute score on incompatible types
                logger.warning(f"Could not compute AUC: {e}")

    # Get feature importance
    feature_importance = None
    try:
        if test_df is not None:
            feature_importance = predictor.feature_importance(test_df)
        else:
            feature_importance = predictor.feature_importance(train_df)
    except (AttributeError, ValueError, RuntimeError) as e:
        # AttributeError: Model doesn't support feature importance
        # ValueError: Invalid input data
        # RuntimeError: Feature importance computation failed
        logger.warning(f"Could not compute feature importance: {e}")

    if verbose:
        logger.info("AutoGluon training completed")

    return SimpleNamespace(
        model=predictor,
        metrics=metrics,
        fi=feature_importance,
        test_target=test_target,
        test_probs=test_probs,
    )


def train_lama_model(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_name: str = "target",
    init_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
) -> Optional[SimpleNamespace]:
    """
    Train a LightAutoML (LAMA) model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame (must include target column).
    test_df : pd.DataFrame, optional
        Test DataFrame for evaluation (must include target column).
    target_name : str, default="target"
        Name of the target column.
    init_params : dict, optional
        Parameters for TabularAutoML initialization. If not provided,
        defaults to binary classification task.
    fit_params : dict, optional
        Parameters for the fit_predict() method.
    verbose : int, default=1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Returns
    -------
    SimpleNamespace or None
        Returns None if LightAutoML is not installed.
        Otherwise returns SimpleNamespace with:
        - model: Trained TabularAutoML instance
        - metrics: dict with evaluation metrics (e.g., test_auc)
        - fi: Feature importance DataFrame
        - test_target: Test target values (if test_df provided)
        - test_probs: Prediction probabilities (if test_df provided)

    Raises
    ------
    ImportError
        If LightAutoML is not installed (returns None instead of raising).

    Notes
    -----
    This function assumes binary classification by default when init_params
    is not provided. For regression or multiclass tasks, pass an appropriate
    Task object in init_params.
    """
    try:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        import matplotlib as mpl
    except ImportError as e:
        logger.error(f"LightAutoML not available: {e}")
        return None

    # Default to binary classification if no init_params provided
    if init_params is None:
        init_params = {"task": Task("binary")}
    fit_params = fit_params or {}

    if verbose:
        logger.info(f"Training LightAutoML model on {len(train_df)} rows...")

    # Create automl
    automl = TabularAutoML(**init_params)

    # Fit model
    automl.fit_predict(train_df, roles={"target": target_name}, verbose=verbose, **fit_params)

    clean_ram()
    if verbose:
        log_ram_usage()

    # Evaluate on test set if provided
    test_probs = None
    test_target = None
    metrics = {}

    if test_df is not None and len(test_df) > 0:
        if target_name in test_df.columns:
            test_target = test_df[target_name]

        test_predictions = automl.predict(test_df)
        # LAMA returns predictions in a specific format (binary classification)
        # Validate array shape before indexing
        if (
            hasattr(test_predictions, "data")
            and test_predictions.data is not None
            and test_predictions.data.ndim >= 1
            and test_predictions.data.shape[-1] >= 1
        ):
            pred_col = test_predictions.data[:, 0] if test_predictions.data.ndim > 1 else test_predictions.data
            test_probs = np.vstack([1 - pred_col, pred_col]).T
        else:
            logger.warning("LAMA predictions have unexpected shape, skipping probability conversion")
            test_probs = None

        if test_target is not None and verbose and test_probs is not None:
            try:
                auc = roc_auc_score(test_target, test_probs[:, 1])
                logger.info(f"LAMA test AUC: {auc:.4f}")
                metrics["test_auc"] = auc
            except (ValueError, TypeError) as e:
                # ValueError: Only one class present in y_true or invalid input
                # TypeError: Cannot compute score on incompatible types
                logger.warning(f"Could not compute AUC: {e}")

        # Reset matplotlib params (LAMA sometimes modifies them)
        mpl.rcParams.update(mpl.rcParamsDefault)

    # Get feature importance
    feature_importance = None
    try:
        feature_importance = automl.get_feature_scores("fast")
    except (AttributeError, ValueError, RuntimeError) as e:
        # AttributeError: Model doesn't support feature importance
        # ValueError: Invalid input data
        # RuntimeError: Feature importance computation failed
        logger.warning(f"Could not compute feature importance: {e}")

    if verbose:
        logger.info("LAMA training completed")

    return SimpleNamespace(
        model=automl,
        metrics=metrics,
        fi=feature_importance,
        test_target=test_target,
        test_probs=test_probs,
    )


def train_automl_models_suite(
    train_df: Union[pd.DataFrame, pl.DataFrame],
    test_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    target_name: str = "target",
    config: Optional[AutoMLConfig] = None,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Train AutoML models (AutoGluon and/or LAMA) on a dataset.

    Parameters
    ----------
    train_df : pd.DataFrame or pl.DataFrame
        Training DataFrame with target column included.
    test_df : pd.DataFrame or pl.DataFrame, optional
        Test DataFrame with target column for evaluation.
    target_name : str, default="target"
        Name of the target column.
    config : AutoMLConfig, optional
        AutoML configuration. If not provided, uses default configuration.
    verbose : int, default=1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Returns
    -------
    dict
        Dictionary mapping model names to SimpleNamespace results:
        - "autogluon": AutoGluon results (if use_autogluon=True)
        - "lama": LightAutoML results (if use_lama=True)

    Raises
    ------
    ValueError
        If train_df is empty or target column is not found.

    Notes
    -----
    - AutoML models require the target to be in the dataframe
    - No separate preprocessing or scaling is needed
    - No validation set is used (AutoML handles internal validation)

    Examples
    --------
    >>> config = AutoMLConfig(
    ...     use_autogluon=True,
    ...     use_lama=False,
    ...     autogluon_init_params=dict(eval_metric='log_loss'),
    ...     autogluon_fit_params=dict(time_limit=3600, presets='best_quality'),
    ... )
    >>> models = train_automl_models_suite(
    ...     train_df=train_df,  # Must include target column
    ...     test_df=test_df,    # Must include target column
    ...     target_name="target",
    ...     config=config,
    ... )
    """
    if config is None:
        config = AutoMLConfig()

    # Convert Polars to Pandas (AutoML libraries require pandas)
    if isinstance(train_df, pl.DataFrame):
        train_df = get_pandas_view_of_polars_df(train_df)
    if test_df is not None and isinstance(test_df, pl.DataFrame):
        test_df = get_pandas_view_of_polars_df(test_df)

    # Validate input DataFrames
    if len(train_df) == 0:
        raise ValueError("train_df is empty, cannot train AutoML models")
    if target_name not in train_df.columns:
        raise ValueError(f"Target column '{target_name}' not found in train_df")
    if test_df is not None and target_name not in test_df.columns:
        raise ValueError(f"Target column '{target_name}' not found in test_df")

    models = {}

    # Train AutoGluon
    if config.use_autogluon:
        if verbose:
            logger.info(LOG_SEPARATOR)
            logger.info("Training AutoGluon model...")
            logger.info(LOG_SEPARATOR)

        ag_result = train_autogluon_model(
            train_df=train_df,
            test_df=test_df,
            target_name=config.automl_target_label or target_name,
            init_params=config.autogluon_init_params,
            fit_params=config.autogluon_fit_params,
            verbose=config.automl_verbose,
        )

        if ag_result is not None:
            models["autogluon"] = ag_result

    # Train LAMA
    if config.use_lama:
        if verbose:
            logger.info(LOG_SEPARATOR)
            logger.info("Training LightAutoML (LAMA) model...")
            logger.info(LOG_SEPARATOR)

        lama_result = train_lama_model(
            train_df=train_df,
            test_df=test_df,
            target_name=config.automl_target_label or target_name,
            init_params=config.lama_init_params,
            fit_params=config.lama_fit_params,
            verbose=config.automl_verbose,
        )

        if lama_result is not None:
            models["lama"] = lama_result

    if verbose:
        logger.info(f"AutoML training suite completed. Trained {len(models)} model(s).")

    return models


__all__ = [
    # Constants
    "LOG_SEPARATOR",
    # Functions
    "train_autogluon_model",
    "train_lama_model",
    "train_automl_models_suite",
]
