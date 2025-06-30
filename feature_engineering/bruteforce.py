"""FE using bruteforce search."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *


import textwrap
import warnings
import numpy as np
import pandas as pd
import polars as pl

from pysr import PySRRegressor

from category_encoders import CatBoostEncoder

from pyutilz.system import clean_ram

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def run_pysr_feature_engineering(
    df: Union[pd.DataFrame, pl.DataFrame],
    target_col: str,
    drop_columns: Optional[List[str]] = None,
    reserved_names: Optional[List[str]] = None,
    reserved_prefix: str = "reserved_",
    sample_size: int = 30_000,
    encode_categoricals: bool = True,
    string_categorical_threshold: int = 100,
    pysr_params: Optional[Dict] = None,
    pysr_params_override: Optional[Dict] = None,
    verbose: int = 1,
) -> PySRRegressor:
    """
    Run symbolic regression on a sampled version of a DataFrame using PySR.

    Parameters:
    - df: Input pandas or polars DataFrame.
    - target_col: Column to use as the regression target.
    - drop_other_targets: Whether to drop other target-like columns (e.g. target_UP, target_DOWN).
    - reserved_names: List of reserved or conflicting column names to rename using a prefix.
    - reserved_prefix: Prefix to add to reserved column names.
    - sample_size: How many rows to sample from df.
    - encode_categoricals: Whether to encode categoricals using CatBoostEncoder or drop them.
    - string_categorical_threshold: Maximum unique values allowed in string columns to treat them as categoricals.
    - drop_columns: List of known columns to drop (e.g. identifiers like 'ts', 'secid').
    - pysr_params: Dict of base PySRRegressor parameters (will be used as default).
    - pysr_params_override: Dict to override values in pysr_params.
    - verbose: Controls output. 0 = silent, >0 = warnings and info messages.

    Returns:
    - Trained PySRRegressor object.

    >>>run_pysr_feature_engineering(df,target_col="target_UP",drop_columns=["target_UP","target_DOWN"],pysr_params_override=dict(niterations=200,))
    """
    if reserved_names is None:
        reserved_names = ["im"]

    # Convert polars to pandas if needed
    if isinstance(df, pl.DataFrame):
        tmp_df = df.sample(sample_size).fill_null(0).fill_nan(0).to_pandas()
    elif isinstance(df, pd.DataFrame):
        tmp_df = df.sample(min(sample_size, len(df))).fillna(0).copy()
    else:
        raise ValueError("Input must be a pandas or polars DataFrame.")

    clean_ram()

    # Sanitize column names
    tmp_df.columns = [col.replace("-", "_").replace("=", "_") for col in tmp_df.columns]

    # Rename reserved names with prefix
    tmp_df.rename(columns={col: reserved_prefix + col for col in reserved_names if col in tmp_df.columns}, inplace=True)

    # Separate target
    if target_col not in tmp_df:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    target = tmp_df[target_col].copy()

    if target_col not in drop_columns:
        drop_columns.append(target_col)

    for col in drop_columns:
        if col in tmp_df:
            del tmp_df[col]

    # Drop or warn about datetime columns
    datetime_cols = tmp_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if datetime_cols and verbose > 0:
        wrapped_names = textwrap.fill(", ".join(datetime_cols), width=80)
        warnings.warn(f"Excluding {len(datetime_cols)} datetime columns: {wrapped_names}")
    tmp_df.drop(columns=datetime_cols, inplace=True)

    # Detect and encode string columns
    str_cols = tmp_df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in str_cols:
        unique_vals = tmp_df[col].nunique()
        if unique_vals <= string_categorical_threshold:
            tmp_df[col] = tmp_df[col].astype("category")
        else:
            if verbose > 0:
                warnings.warn(f"Dropping string column '{col}' with {unique_vals} unique values.")
            tmp_df.drop(columns=[col], inplace=True)

    # Encode or drop categoricals
    cat_cols = tmp_df.select_dtypes(include=["category"]).columns.tolist()
    if encode_categoricals and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, return_df=True)
        tmp_df[cat_cols] = encoder.fit_transform(tmp_df[cat_cols], target)
        if verbose > 0:
            warnings.warn(f"Encoded categorical columns using CatBoostEncoder: {cat_cols}")
    elif not encode_categoricals and cat_cols:
        if verbose > 0:
            warnings.warn(f"Dropping categorical columns: {cat_cols}")
        tmp_df.drop(columns=cat_cols, inplace=True)

    # Drop remaining non-numeric columns
    non_numeric = tmp_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        if verbose > 0:
            warnings.warn(f"Dropping non-numeric columns: {non_numeric}")
        tmp_df.drop(columns=non_numeric, inplace=True)

    clean_ram()

    # Define default PySR parameters
    default_params = dict(
        maxsize=14,
        niterations=2000,
        binary_operators=["+", "*"],
        unary_operators=["log", "inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        turbo=True,
        bumper=True,
    )

    final_params = pysr_params.copy() if pysr_params else default_params
    if pysr_params_override:
        final_params.update(pysr_params_override)

    fe_model = PySRRegressor(**final_params)

    fe_model.fit(tmp_df, target)
    clean_ram()

    if verbose > 0:
        print("Best equation:")
        print(fe_model.get_best())
        print("\nAll equations:")
        print(fe_model.equations.equation.tolist())

    return fe_model
