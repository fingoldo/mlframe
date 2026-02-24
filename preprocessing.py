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

import pandas as pd, polars as pl
from pyutilz.system import tqdmu


def prepare_df_for_catboost(
    df: object,
    columns_to_drop: Sequence = [],
    text_features: Sequence = [],
    cat_features: list = [],
    na_filler: str = "",
    ensure_categorical: bool = True,
    verbose: bool = False,
):
    """
    Catboost needs NAs in cat features replaced by a string value.
    Possibly extends cat_features list.
    ensure_categorical:bool=True makes further processing also suitable for xgboost.
    Works with both pandas and polars DataFrames. Always use the return value.
    """
    is_polars = isinstance(df, pl.DataFrame)

    if columns_to_drop:
        if is_polars:
            df = df.drop([c for c in columns_to_drop if c in df.columns])
        else:
            df.drop(columns=columns_to_drop, inplace=True)

    cols = set(df.columns)

    if is_polars:
        # Text features: fill nulls
        text_exprs = []
        for var in tqdmu(text_features, desc="Processing textual features for CatBoost...", leave=False):
            if var in cols and df[var].is_null().any():
                text_exprs.append(pl.col(var).fill_null(na_filler))
        if text_exprs:
            df = df.with_columns(text_exprs)

        # Cast nullable integer/boolean columns to Float64 so that to_pandas() produces
        # numpy float64 with np.nan instead of nullable Int64/boolean with pd.NA,
        # which CatBoost cannot handle.
        _NULLABLE_INT_BOOL = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Boolean)
        numeric_exprs = []
        for var in cols:
            if var not in cat_features and var not in text_features:
                dtype = df[var].dtype
                if dtype in _NULLABLE_INT_BOOL and df[var].is_null().any():
                    numeric_exprs.append(pl.col(var).cast(pl.Float64))
        if numeric_exprs:
            df = df.with_columns(numeric_exprs)

        # Categorical features
        cat_exprs = []
        for var in tqdmu(cols, desc="Processing categorical features for CatBoost...", leave=False):
            dtype = df[var].dtype
            is_cat = dtype == pl.Categorical or isinstance(dtype, pl.Enum)
            if is_cat:
                if df[var].is_null().any():
                    cat_exprs.append(pl.col(var).cast(pl.String).fill_null(na_filler).cast(pl.Categorical))
                if var not in cat_features:
                    if verbose:
                        logging.info(f"{var} appended to cat_features")
                    cat_features.append(var)
            elif var in cat_features:
                expr = pl.col(var)
                if df[var].is_null().any():
                    expr = expr.fill_null(na_filler)
                if ensure_categorical:
                    try:
                        expr = expr.cast(pl.Categorical)
                    except Exception:
                        logger.warning(f"Could not convert column {var} to categorical.")
                        expr = None
                if expr is not None:
                    cat_exprs.append(expr)
        if cat_exprs:
            df = df.with_columns(cat_exprs)
    else:
        for var in tqdmu(text_features, desc="Processing textual features for CatBoost...", leave=False):
            if var in cols:
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
                        try:
                            df[var] = df[var].astype("category")
                        except Exception:
                            logger.warning(f"Could not convert column {var} to categorical.")
                elif pd.api.types.is_extension_array_dtype(df[var].dtype):
                    # Nullable extension dtypes (Int64, Float64, boolean, etc.) use pd.NA,
                    # which CatBoost cannot handle — convert to numpy float64 (pd.NA → np.nan)
                    try:
                        df[var] = df[var].astype(float)
                    except Exception:
                        pass

    return df


def prepare_df_for_xgboost(
    df: object,
    cat_features: Sequence = [],
    ensure_categorical: bool = True,
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


def pack_val_set_into_fit_params(model: object, X_val: pd.DataFrame, y_val: pd.DataFrame, early_stopping_rounds: int, cat_features: list = None) -> dict:
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
