"""
AutoML model training for mlframe.

Separate module for AutoGluon and LightAutoML (LAMA) training.
These models require the target to be in the dataframe, so they work differently
from the regular mlframe pipeline.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
from types import SimpleNamespace
from pyutilz.system import clean_ram

from .configs import AutoMLConfig
from .utils import log_ram_usage, get_pandas_view_of_polars_df

logger = logging.getLogger(__name__)


def train_autogluon_model(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_name: str = "target",
    init_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
) -> SimpleNamespace:
    """
    Train an AutoGluon model.

    Args:
        train_df: Training DataFrame (must include target column)
        test_df: Test DataFrame (optional, must include target column)
        target_name: Name of target column
        init_params: Parameters for TabularPredictor initialization
        fit_params: Parameters for fit() method
        verbose: Verbosity level

    Returns:
        SimpleNamespace with model, metrics, feature_importance, test_probs
    """
    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
    except ImportError as e:
        logger.error(f"AutoGluon not available: {e}")
        return None

    if init_params is None:
        init_params = {}
    if fit_params is None:
        fit_params = {}

    if verbose:
        logger.info(f"Training AutoGluon model on {len(train_df)} rows...")

    # Create predictor
    predictor = TabularPredictor(
        label=target_name,
        verbosity=verbose,
        **init_params
    )

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
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(test_target, test_probs[:, 1])
                logger.info(f"AutoGluon test AUC: {auc:.4f}")
                metrics['test_auc'] = auc
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")

    # Get feature importance
    feature_importance = None
    try:
        if test_df is not None:
            feature_importance = predictor.feature_importance(test_df)
        else:
            feature_importance = predictor.feature_importance(train_df)
    except Exception as e:
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
) -> SimpleNamespace:
    """
    Train a LightAutoML (LAMA) model.

    Args:
        train_df: Training DataFrame (must include target column)
        test_df: Test DataFrame (optional, must include target column)
        target_name: Name of target column
        init_params: Parameters for TabularAutoML initialization
        fit_params: Parameters for fit_predict() method
        verbose: Verbosity level

    Returns:
        SimpleNamespace with model, metrics, feature_importance, test_probs
    """
    try:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        import matplotlib as mpl
    except ImportError as e:
        logger.error(f"LightAutoML not available: {e}")
        return None

    if init_params is None:
        # Default to binary classification
        init_params = {'task': Task('binary')}
    if fit_params is None:
        fit_params = {}

    if verbose:
        logger.info(f"Training LightAutoML model on {len(train_df)} rows...")

    # Create automl
    automl = TabularAutoML(**init_params)

    # Fit model
    out_of_fold_predictions = automl.fit_predict(
        train_df,
        roles={'target': target_name},
        verbose=verbose,
        **fit_params
    )

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
        # LAMA returns predictions in a specific format
        test_probs = np.vstack([1 - test_predictions.data[:, 0], test_predictions.data[:, 0]]).T

        if test_target is not None and verbose:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(test_target, test_probs[:, 1])
                logger.info(f"LAMA test AUC: {auc:.4f}")
                metrics['test_auc'] = auc
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")

        # Reset matplotlib params (LAMA sometimes modifies them)
        mpl.rcParams.update(mpl.rcParamsDefault)

    # Get feature importance
    feature_importance = None
    try:
        feature_importance = automl.get_feature_scores("fast")
    except Exception as e:
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
    train_df: Union[pd.DataFrame, 'pl.DataFrame'],
    test_df: Optional[Union[pd.DataFrame, 'pl.DataFrame']] = None,
    target_name: str = "target",
    config: Optional[AutoMLConfig] = None,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Train AutoML models (AutoGluon and/or LAMA) on a dataset.

    Args:
        train_df: Training DataFrame with target column
        test_df: Test DataFrame with target column (optional)
        target_name: Name of the target column
        config: AutoML configuration
        verbose: Verbosity level

    Returns:
        Dictionary of trained models and results

    Notes:
        - AutoML models require the target to be in the dataframe
        - No separate preprocessing or scaling is needed
        - No validation set is used (AutoML handles internal validation)

    Example:
        ```python
        config = AutoMLConfig(
            use_autogluon=True,
            use_lama=False,
            autogluon_init_params=dict(eval_metric='log_loss'),
            autogluon_fit_params=dict(time_limit=3600, presets='best_quality'),
        )

        models = train_automl_models_suite(
            train_df=train_df,  # Must include target column
            test_df=test_df,    # Must include target column
            target_name="target",
            config=config,
        )
        ```
    """
    import polars as pl

    if config is None:
        config = AutoMLConfig()

    # Convert Polars to Pandas (AutoML libraries require pandas)
    if isinstance(train_df, pl.DataFrame):
        train_df = get_pandas_view_of_polars_df(train_df).to_pandas()
    if test_df is not None and isinstance(test_df, pl.DataFrame):
        test_df = get_pandas_view_of_polars_df(test_df).to_pandas()

    # Validate target column
    if target_name not in train_df.columns:
        raise ValueError(f"Target column '{target_name}' not found in train_df")
    if test_df is not None and target_name not in test_df.columns:
        raise ValueError(f"Target column '{target_name}' not found in test_df")

    models = {}

    # Train AutoGluon
    if config.use_autogluon:
        if verbose:
            logger.info("="*80)
            logger.info("Training AutoGluon model...")
            logger.info("="*80)

        ag_result = train_autogluon_model(
            train_df=train_df,
            test_df=test_df,
            target_name=config.automl_target_label or target_name,
            init_params=config.autogluon_init_params,
            fit_params=config.autogluon_fit_params,
            verbose=config.automl_verbose,
        )

        if ag_result is not None:
            models['autogluon'] = ag_result

    # Train LAMA
    if config.use_lama:
        if verbose:
            logger.info("="*80)
            logger.info("Training LightAutoML (LAMA) model...")
            logger.info("="*80)

        lama_result = train_lama_model(
            train_df=train_df,
            test_df=test_df,
            target_name=config.automl_target_label or target_name,
            init_params=config.lama_init_params,
            fit_params=config.lama_fit_params,
            verbose=config.automl_verbose,
        )

        if lama_result is not None:
            models['lama'] = lama_result

    if verbose:
        logger.info(f"AutoML training suite completed. Trained {len(models)} model(s).")

    return models


__all__ = [
    'train_autogluon_model',
    'train_lama_model',
    'train_automl_models_suite',
]
