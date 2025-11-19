"""
Model evaluation and reporting functions for mlframe training.

For now, this module re-exports functions from the main training module.
In the future, these can be fully refactored into this module.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Callable, Union
from sklearn.base import ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)

# Import existing evaluation functions from main training module
# This allows us to refactor incrementally without breaking functionality
try:
    from ..training import (
        report_model_perf,
        report_regression_model_perf,
        report_probabilistic_model_perf,
        get_model_feature_importances,
        plot_model_feature_importances,
        post_calibrate_model,
    )

    _IMPORTED_FROM_MAIN = True
except ImportError:
    _IMPORTED_FROM_MAIN = False
    logger.warning("Could not import evaluation functions from main training module")


def evaluate_model(
    model: Union[ClassifierMixin, RegressorMixin],
    model_name: str,
    targets: Union[np.ndarray, pd.Series],
    columns: Sequence,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    show_fi: bool = True,
    verbose: int = 1,
    **kwargs,
) -> tuple:
    """
    Evaluate a trained model and generate reports.

    Args:
        model: Trained model
        model_name: Name for reporting
        targets: True target values
        columns: Feature column names
        preds: Predictions (optional, will be generated if not provided)
        probs: Probabilities for classification (optional)
        df: DataFrame with features (optional)
        show_fi: Whether to show feature importances
        verbose: Verbosity level
        **kwargs: Additional arguments passed to report functions

    Returns:
        Tuple of (preds, probs) or (preds, None) for regression
    """
    if _IMPORTED_FROM_MAIN:
        return report_model_perf(
            targets=targets,
            columns=columns,
            model_name=model_name,
            model=model,
            preds=preds,
            probs=probs,
            df=df,
            show_fi=show_fi,
            **kwargs,
        )
    else:
        # Fallback: minimal evaluation
        if preds is None and model is not None:
            if df is not None:
                preds = model.predict(df)
            else:
                raise ValueError("Either preds or df must be provided")

        if probs is None and hasattr(model, 'predict_proba'):
            if df is not None:
                probs = model.predict_proba(df)

        if verbose:
            from sklearn.metrics import accuracy_score, mean_squared_error
            if probs is not None:
                acc = accuracy_score(targets, preds)
                logger.info(f"{model_name}: Accuracy = {acc:.4f}")
            else:
                mse = mean_squared_error(targets, preds)
                logger.info(f"{model_name}: MSE = {mse:.4f}")

        return preds, probs


# Re-export for convenience
__all__ = [
    'evaluate_model',
]

# Add conditional exports if imported successfully
if _IMPORTED_FROM_MAIN:
    __all__.extend([
        'report_model_perf',
        'report_regression_model_perf',
        'report_probabilistic_model_perf',
        'get_model_feature_importances',
        'plot_model_feature_importances',
        'post_calibrate_model',
    ])
