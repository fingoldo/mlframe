"""FE using bruteforce symbolic-regression search via PySR."""

from __future__ import annotations


__all__ = [
    "DEFAULT_BINARY_OPERATORS",
    "DEFAULT_UNARY_OPERATORS",
    "run_pysr_feature_engineering",
]

import logging
import textwrap
import threading
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import clean_ram

logger = logging.getLogger(__name__)

# Module-level lock serialises Julia access across concurrent joblib workers. PySR spawns
# Julia subprocesses that contend for the same .julia/ precompile cache and shared memory;
# the lock keeps them sequential per-process.
_PYSR_LOCK = threading.Lock()

DEFAULT_BINARY_OPERATORS: List[str] = ["+", "*"]
DEFAULT_UNARY_OPERATORS: List[str] = ["log", "inv(x) = 1/x"]
# Cap expression complexity: maxsize=14 keeps trees human-readable while still discovering ratios/products of 3-4 features.
_DEFAULT_PYSR_MAXSIZE = 14
# 2000 iterations is the empirical knee where PySR loss plateaus on tabular FE search; more iterations rarely improve the Pareto front but linearly cost Julia time.
_DEFAULT_PYSR_NITERATIONS = 2000
# 30k rows is enough for stable symbolic-regression fitness on tabular data; larger samples dominate Julia runtime without changing the discovered expressions.
_DEFAULT_SAMPLE_SIZE = 30_000


def _kfold_target_encode(
    df: pd.DataFrame,
    cols: List[str],
    target: pd.Series,
    n_splits: int = 5,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Out-of-fold CatBoostEncoder target encoding.

    Each row's encoded value is computed from a fold of which that row is NOT a member, so the
    target never leaks into its own feature. Equivalent to ``cross_val_predict``-style OOF
    encoding. Used when the caller asks for ``leakage_free=True``.
    """
    from category_encoders import CatBoostEncoder
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    encoded = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        encoder = CatBoostEncoder(cols=cols, return_df=True)
        encoder.fit(df.iloc[train_idx][cols], target.iloc[train_idx])
        encoded.iloc[val_idx, :] = encoder.transform(df.iloc[val_idx][cols]).values
        logger.debug("Fold %d/%d encoded.", fold_idx + 1, n_splits)
    return encoded


def run_pysr_feature_engineering(
    df: Union[pd.DataFrame, pl.DataFrame],
    target_col: str,
    drop_columns: Optional[List[str]] = None,
    reserved_names: Optional[List[str]] = None,
    reserved_prefix: str = "reserved_",
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    encode_categoricals: bool = True,
    string_categorical_threshold: int = 100,
    pysr_params: Optional[Dict] = None,
    pysr_params_override: Optional[Dict] = None,
    leakage_free: bool = True,
    leakage_free_n_splits: int = 5,
    random_state: Optional[int] = None,
    verbose: int = 1,
):
    """Run symbolic regression on a sampled frame using PySR.

    Parameters
    ----------
    df
        Input pandas or polars DataFrame.
    target_col
        Column to use as the regression target.
    drop_columns
        Columns to drop from the feature matrix (other target columns, identifiers like
        ``ts`` / ``secid``, etc.). The caller's list is never mutated.
    reserved_names
        Column names that would clash with PySR-reserved identifiers (e.g. ``im``); they get
        renamed with ``reserved_prefix``. Defaults to ``["im"]``.
    reserved_prefix
        Prefix applied to ``reserved_names``.
    sample_size
        How many rows to sample from ``df``.
    encode_categoricals
        ``True`` -> encode categorical columns with CatBoostEncoder; ``False`` -> drop them.
    string_categorical_threshold
        Maximum unique values allowed in string columns to treat them as categoricals.
    pysr_params
        Dict merged on top of the module defaults.
    pysr_params_override
        Dict applied last; wins over both defaults and ``pysr_params``.
    leakage_free
        ``True`` (default) runs CatBoostEncoder in OOF/KFold mode (no row sees its own target). Set ``False``
        only for legacy / quick-look exploratory runs where holdout-honesty is not required; a warning fires
        in that case because downstream metrics will be optimistically biased.
    leakage_free_n_splits
        Number of folds when ``leakage_free=True``.
    random_state
        Forwarded to ``.sample(...)`` and the KFold splitter for reproducibility.
    verbose
        ``0`` = silent; ``>0`` = log info + warn about excluded columns.

    Returns
    -------
    Trained PySRRegressor instance.

    Examples
    --------
    >>> run_pysr_feature_engineering(  # doctest: +SKIP
    ...     df,
    ...     target_col="target_UP",
    ...     drop_columns=["target_UP", "target_DOWN"],
    ...     pysr_params_override=dict(niterations=200),
    ... )
    """
    # Lazy heavy imports: importing this module shouldn't pay PySR/Julia startup cost.
    from pysr import PySRRegressor

    if reserved_names is None:
        reserved_names = ["im"]

    if isinstance(df, pl.DataFrame):
        import polars.selectors as cs
        n = min(sample_size, len(df))
        sampled = df.sample(n, seed=random_state) if random_state is not None else df.sample(n)
        # Zero-fill ONLY numeric columns; previously a blanket .fill_null(0) ran before non-numeric
        # columns were dropped, which raises on polars 1.x for Utf8 / Datetime / Duration dtypes.
        # Wave 50 (2026-05-20): use per-column median instead of 0 so NaN rows aren't
        # silently collapsed onto real-0 rows in PySR's candidate-score ranking. Median
        # is robust to outliers and preserves the column's central tendency.
        sampled = sampled.with_columns([
            cs.numeric().fill_nan(cs.numeric().median()).fill_null(cs.numeric().median())
        ])
        tmp_df = sampled.to_pandas()
    elif isinstance(df, pd.DataFrame):
        n = min(sample_size, len(df))
        tmp_df = df.sample(n, random_state=random_state)
        # Zero-fill ONLY numeric columns; ``.fillna(0)`` on a Categorical with a NaN raises
        # ``Cannot setitem on a Categorical with a new category (0)`` because ``0`` is not a
        # listed category. Categoricals get dropped or encoded downstream anyway, so we leave
        # their NaNs alone here.
        # Wave 50 (2026-05-20): use per-column median for the same reason as the polars branch.
        numeric_cols = tmp_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            tmp_df[numeric_cols] = tmp_df[numeric_cols].fillna(tmp_df[numeric_cols].median())
    else:
        raise TypeError(f"Input must be a pandas or polars DataFrame, got {type(df).__name__}.")

    clean_ram()

    tmp_df.columns = [col.replace("-", "_").replace("=", "_") for col in tmp_df.columns]
    tmp_df.rename(
        columns={col: reserved_prefix + col for col in reserved_names if col in tmp_df.columns},
        inplace=True,
    )

    if target_col not in tmp_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in dataframe.")

    target = tmp_df[target_col].copy()

    # Work on a local copy so the caller's drop_columns list is not mutated.
    drop_set = set(drop_columns or [])
    drop_set.add(target_col)
    cols_to_drop = [c for c in drop_set if c in tmp_df.columns]
    tmp_df.drop(columns=cols_to_drop, inplace=True)

    datetime_cols = tmp_df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols and verbose > 0:
        wrapped_names = textwrap.fill(", ".join(datetime_cols), width=80)
        logger.info("Excluding %d datetime columns: %s", len(datetime_cols), wrapped_names)
    tmp_df.drop(columns=datetime_cols, inplace=True)

    str_cols = tmp_df.select_dtypes(include=["object", "string"]).columns.tolist()
    cols_to_drop_str: List[str] = []
    for col in str_cols:
        unique_vals = tmp_df[col].nunique()
        if unique_vals <= string_categorical_threshold:
            tmp_df[col] = tmp_df[col].astype("category")
        else:
            cols_to_drop_str.append(col)
            if verbose > 0:
                logger.info("Dropping string column %r with %d unique values.", col, unique_vals)
    if cols_to_drop_str:
        tmp_df.drop(columns=cols_to_drop_str, inplace=True)

    cat_cols = tmp_df.select_dtypes(include=["category"]).columns.tolist()
    if encode_categoricals and cat_cols:
        if leakage_free:
            logger.info(
                "CatBoostEncoder running in OOF/KFold mode (%d splits) over %d categorical columns: %s",
                leakage_free_n_splits,
                len(cat_cols),
                cat_cols,
            )
            tmp_df[cat_cols] = _kfold_target_encode(
                tmp_df, cat_cols, target,
                n_splits=leakage_free_n_splits,
                random_state=random_state,
            )
        else:
            # The legacy path fits CatBoostEncoder on the full sample with the target visible -
            # classic supervised-encoding leak. Surface as a warning so the operator can switch
            # to leakage_free=True (slower, but holdout-honest).
            warnings.warn(
                "run_pysr_feature_engineering: CatBoostEncoder.fit_transform on the full sample "
                "leaks the target into the categorical encoding; downstream holdout metrics will "
                f"be optimistically biased. Columns: {cat_cols}. Pass leakage_free=True for an "
                "OOF-encoded path.",
                stacklevel=2,
            )
            from category_encoders import CatBoostEncoder

            encoder = CatBoostEncoder(cols=cat_cols, return_df=True)
            tmp_df[cat_cols] = encoder.fit_transform(tmp_df[cat_cols], target)
        if verbose > 0:
            logger.info("Encoded %d categorical column(s) via CatBoostEncoder.", len(cat_cols))
    elif not encode_categoricals and cat_cols:
        if verbose > 0:
            logger.info("Dropping categorical columns: %s", cat_cols)
        tmp_df.drop(columns=cat_cols, inplace=True)

    non_numeric = tmp_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        if verbose > 0:
            logger.info("Dropping non-numeric columns: %s", non_numeric)
        tmp_df.drop(columns=non_numeric, inplace=True)

    clean_ram()

    default_params: Dict = dict(
        maxsize=_DEFAULT_PYSR_MAXSIZE,
        niterations=_DEFAULT_PYSR_NITERATIONS,
        binary_operators=list(DEFAULT_BINARY_OPERATORS),
        unary_operators=list(DEFAULT_UNARY_OPERATORS),
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
        turbo=True,
        bumper=True,
    )

    final_params = {**default_params, **(pysr_params or {})}
    if pysr_params_override:
        final_params.update(pysr_params_override)

    fe_model = PySRRegressor(**final_params)

    with _PYSR_LOCK:
        fe_model.fit(tmp_df, target)
    clean_ram()

    if verbose > 0:
        logger.info("Best equation:\n%s", fe_model.get_best())
        logger.info("All equations:\n%s", fe_model.equations.equation.tolist())

    return fe_model
