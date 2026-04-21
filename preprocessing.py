# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from .config import XGBOOST_MODEL_TYPES, LGBM_MODEL_TYPES, CATBOOST_MODEL_TYPES

import numpy as np, pandas as pd
from pyutilz.system import tqdmu


def _get_polars():
    """Lazy polars import — heavy native deps, not every caller needs it."""
    import polars as pl
    return pl


def prepare_df_for_catboost(
    df: object,
    columns_to_drop: Optional[Sequence] = None,
    text_features: Optional[Sequence] = None,
    cat_features: Optional[list] = None,
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
    # Avoid the classic mutable-default-argument bug: a list literal in the
    # signature is shared across calls, so every .append() leaked across
    # invocations. Accept None and normalize here.
    if columns_to_drop is None:
        columns_to_drop = []
    if cat_features is None:
        cat_features = []

    try:
        pl = _get_polars()
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        pl = None
        is_polars = False

    if text_features is None:
        text_features = []

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

        # Cast nullable integer/boolean columns to Float* so that to_pandas() produces
        # numpy floats with np.nan instead of nullable Int*/boolean with pd.NA, which
        # CatBoost cannot handle. Preserve precision — Int8/16/32 and Boolean fit
        # exactly into Float32, only Int64/UInt64 need Float64 to avoid silent loss
        # beyond ~2**24.
        _INT_BOOL_TO_F32 = (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.Boolean)
        _INT_TO_F64 = (pl.Int64, pl.UInt64)
        numeric_exprs = []
        for var in cols:
            if var not in cat_features and var not in text_features:
                dtype = df[var].dtype
                if df[var].is_null().any():
                    if dtype in _INT_BOOL_TO_F32:
                        numeric_exprs.append(pl.col(var).cast(pl.Float32))
                    elif dtype in _INT_TO_F64:
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

        # Columns declared as text_features must bypass the cat-preparation
        # path entirely. Without this skip, a text column that happens to
        # carry a pd.Categorical dtype (e.g. auto-promoted from cat to text,
        # then converted from Polars Categorical to pandas) triggers the
        # O(n_rows) astype(str).astype("category") rebuild below — which on
        # a high-cardinality text blob (81k unique values over 810k rows)
        # takes minutes per column and was the root of a hang observed
        # 2026-04-19 in the CB pandas fallback path.
        text_feature_set = set(text_features or [])
        for var in tqdmu(cols, desc="Processing categorical features for CatBoost...", leave=False):
            if var in text_feature_set:
                continue
            if isinstance(df[var].dtype, pd.CategoricalDtype):
                if df[var].isna().any():
                    # CRITICAL: never do ``astype(str)`` on a Categorical to
                    # fill NaN. pandas materializes ``categories._values`` as
                    # a fixed-width Unicode array sized by
                    # ``len(categories) × max_str_len × 4``. On columns where
                    # Polars passed through an untrimmed global string-pool
                    # dictionary (3.3M unique categories, 6133-char longest
                    # string seen in prod 2026-04-19), that's a 75+ GiB
                    # allocation → MemoryError.
                    #
                    # Instead, operate on the integer codes: add ``na_filler``
                    # to the category list (O(1) dict growth) and fillna
                    # (O(n_rows) code update, no string materialization).
                    cats = df[var].cat.categories
                    if na_filler not in cats:
                        df[var] = df[var].cat.add_categories([na_filler])
                    df[var] = df[var].fillna(na_filler)
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
                    # which CatBoost cannot handle — convert to numpy floats so pd.NA → np.nan.
                    # Preserve precision: Float32Dtype must stay float32 (callers explicitly
                    # chose narrow precision for memory/GPU); Int8/16/32 and Boolean fit
                    # exactly into float32, only Int64/UInt64/Float64 need float64.
                    try:
                        src = df[var].dtype
                        if isinstance(src, pd.Float32Dtype):
                            target = np.float32
                        elif isinstance(src, pd.Float64Dtype):
                            target = np.float64
                        elif isinstance(src, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype,
                                              pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype,
                                              pd.BooleanDtype)):
                            target = np.float32
                        else:
                            # Int64 / UInt64 / unknown extension: widen to float64.
                            target = np.float64
                        df[var] = df[var].astype(target)
                    except Exception:
                        pass

    return df


def prepare_df_for_xgboost(
    df: pd.DataFrame,
    cat_features: Optional[Sequence] = None,
    ensure_categorical: bool = True,
) -> pd.DataFrame:
    """Ensure categorical columns are pd.CategoricalDtype for XGBoost.

    Contract (fixed 2026-04-19 round 9):
      - Input must be a pandas DataFrame. Polars is NOT accepted; caller
        must convert first (use ``get_pandas_view_of_polars_df``). Pre-fix
        the signature declared ``df: object`` and the function crashed
        with AttributeError on polars input — misleading silent failure.
      - ``cat_features`` is mutated in place (existing contract, kept for
        back-compat) AND the DataFrame is also returned so callers can
        chain calls symmetrically with ``prepare_df_for_catboost``.
      - ``cat_features=None`` is now accepted (coerced to empty list);
        pre-fix hit TypeError on ``var not in None``.

    Args:
        df: pandas DataFrame (polars is rejected with a clear TypeError).
        cat_features: List of known categorical column names, MUTATED in
            place to append auto-detected pd.CategoricalDtype columns.
        ensure_categorical: If True, cast any column in cat_features that
            isn't yet pd.CategoricalDtype to ``category`` dtype.

    Returns:
        The same DataFrame (modified in place for dtype casts).

    Raises:
        TypeError: if ``df`` is a Polars DataFrame or Series.
    """
    try:
        pl = _get_polars()
        is_polars_df = isinstance(df, (pl.DataFrame, pl.Series))
    except ImportError:
        is_polars_df = False
    if is_polars_df:
        raise TypeError(
            f"prepare_df_for_xgboost requires a pandas DataFrame, got "
            f"{type(df).__name__}. Convert via get_pandas_view_of_polars_df() first."
        )
    if cat_features is None:
        cat_features = []
    cols = set(df.columns)
    for var in tqdmu(cols, desc="Processing categorical features for XGBoost...", leave=False):
        if isinstance(df[var].dtype, pd.CategoricalDtype):
            if var not in cat_features:
                logger.info(f"{var} appended to cat_features")
                cat_features.append(var)
        else:
            if var in cat_features and ensure_categorical:
                df[var] = df[var].astype("category")
    return df


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
