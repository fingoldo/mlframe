# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from .config import *

import pandas as pd
from pyutilz.system import tqdmu


def prepare_df_for_catboost(
    df: object,
    columns_to_drop: Sequence = [],
    text_features: Sequence = [],
    cat_features: list = [],
    na_filler: str = "",
    ensure_categorical: bool = True,
    verbose: bool = False,
) -> None:
    """
    Catboost needs NAs in cat features replaced by a string value.
    Possibly extends cat_features list.
    ensure_categorical:bool=True makes further processing also suitable for xgboost.
    """
    cols = set(df.columns)

    for var in tqdmu(text_features, desc="Processing textual features for CatBoost...", leave=False):
        if var in cols and var not in columns_to_drop:
            if df[var].isna().any():
                df[var] = df[var].fillna(na_filler)

    for var in tqdmu(cols, desc="Processing categorical features for CatBoost...", leave=False):
        if isinstance(df[var].dtype, pd.CategoricalDtype):
            if df[var].isna().any():
                df[var] = df[var].astype(str).fillna(na_filler).astype("category")
            if var not in cat_features:
                if verbose:
                    logging.info(f"{var} appended to cat_features")
                cat_features.append(var)
        else:
            if var in cat_features:
                if df[var].isna().any():
                    df[var] = df[var].fillna(na_filler)
                if ensure_categorical:
                    df[var] = df[var].astype("category")


def prepare_df_for_xgboost(
    df: object,
    cat_features: Sequence = [],
    ensure_categorical:bool=True,
) -> None:
    """
    Xgboost needs categorical be of category dtype.
    """
    cols = set(df.columns)
    for var in tqdmu(cols, desc="Processing categorical features for XGBoost...", leave=False):
        if isinstance(df[var].dtype, pd.CategoricalDtype):
            if var not in cat_features:
                logging.info(f"{var} appended to cat_features")
                # df[var] = df[var].astype(str) #(?)
                cat_features.append(var)
        else:
            if var in cat_features and ensure_categorical:
                df[var] = df[var].astype("category")


def pack_val_set_into_fit_params(model: object, X_val: pd.DataFrame, y_val: pd.DataFrame,early_stopping_rounds:int,cat_features:list=None) -> dict:
    """Crafts fir params with early stopping tailored to particular model type."""

    model_type_name = type(model).__name__
    fit_params: dict = {}

    if model_type_name in XGBOOST_MODEL_TYPES:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        fit_params["eval_set"] = ((X_val, y_val),)
    elif model_type_name in LGBM_MODEL_TYPES:
        import lightgbm as lgb

        fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        fit_params["eval_set"] = (X_val, y_val)
    elif model_type_name in CATBOOST_MODEL_TYPES:
        fit_params["use_best_model"] = True
        fit_params["eval_set"] = X_val, y_val
        fit_params["early_stopping_rounds"] = early_stopping_rounds
        if cat_features:
            fit_params["cat_features"] = cat_features
    else:
        raise ValueError(f"eval_set params not known for estimator type: {model}")

    return fit_params
