"""Basic feature engineering for ML."""

from __future__ import annotations


__all__ = [
    "create_date_features",
    "run_pysr_fe",
]

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from pyutilz.system import clean_ram, get_own_memory_usage

logger = logging.getLogger(__name__)

# Map of numpy integer dtypes to polars equivalents. Defined at module scope so it's not
# rebuilt per call. Add new entries here if a user passes a wider integer dtype.
_NP_TO_PL_DTYPE: Dict[type, pl.DataType] = {
    np.int8: pl.Int8,
    np.int16: pl.Int16,
    np.int32: pl.Int32,
    np.int64: pl.Int64,
    np.uint8: pl.UInt8,
    np.uint16: pl.UInt16,
    np.uint32: pl.UInt32,
    np.uint64: pl.UInt64,
}

_DEFAULT_DATE_METHODS: Dict[str, type] = {"day": np.int8, "weekday": np.int8, "month": np.int8}


def create_date_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    cols: List[str],
    delete_original_cols: bool = True,
    methods: Optional[Dict[str, type]] = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Decompose datetime columns into integer date parts (day, weekday, month, ...).

    Parameters
    ----------
    df
        pandas or polars frame to augment.
    cols
        Datetime column names to decompose. Empty list returns ``df`` unchanged.
    delete_original_cols
        Drop the source datetime columns after decomposition.
    methods
        Mapping of ``dt`` accessor name to target numpy integer dtype. Defaults to
        ``{"day": int8, "weekday": int8, "month": int8}``. Weekday is normalised to
        pandas convention (Mon=0 .. Sun=6) in both backends.

    Returns
    -------
    A new frame (same backend as input). The input is never mutated.
    """
    if methods is None:
        # Copy so callers (or future internal mutations) cannot corrupt the module-level singleton; downstream replay maps at `_phase_helpers_fit_pipeline.py` rely on stable iteration order.
        methods = dict(_DEFAULT_DATE_METHODS)
    if len(cols) == 0:
        return df

    logger.debug("In create_date_features. RAM usage: %.1fGB.", get_own_memory_usage())

    is_pandas = isinstance(df, pd.DataFrame)
    is_polars = isinstance(df, pl.DataFrame)
    if not (is_pandas or is_polars):
        raise ValueError("df must be pandas or polars DataFrame")

    # If a derived name (e.g. `date_year`) already exists, with_columns/assign would silently
    # overwrite the caller's feature. Warn once and let the function proceed - raising would
    # regress legitimate re-runs on the same frame.
    existing_cols = set(df.columns)
    derived_names = [f"{col}_{method}" for col in cols for method in methods.keys()]
    clashes = [n for n in derived_names if n in existing_cols]
    if clashes:
        logger.warning(
            "create_date_features: %d derived column name(s) already exist in the DataFrame and will be OVERWRITTEN: %s. "
            "If any of these are user-engineered features, rename either them or the ``methods`` dict keys to disambiguate.",
            len(clashes),
            clashes,
        )

    if is_pandas:
        # Previously mutated the caller's frame in place; the polars branch returned a new frame.
        # Shallow-copy so both branches return a new frame.
        df = df.copy(deep=False)
        for col in cols:
            obj = df[col].dt
            for method, dtype in methods.items():
                if not hasattr(obj, method):
                    raise ValueError(f"Unknown pandas .dt accessor: {method!r}")
                df[col + "_" + method] = getattr(obj, method).astype(dtype)
        if delete_original_cols:
            df = df.drop(columns=cols)
    else:
        all_exprs = []
        for col in cols:
            for method, np_dtype in methods.items():
                pl_dtype = _NP_TO_PL_DTYPE.get(np_dtype)
                if pl_dtype is None:
                    raise ValueError(
                        f"Unsupported dtype {np_dtype} for polars; supported: {list(_NP_TO_PL_DTYPE.keys())}"
                    )

                if method == "weekday":
                    # polars dt.weekday() returns 1..7 (Mon..Sun); subtract 1 to match pandas 0..6.
                    e = (pl.col(col).dt.weekday() - 1).cast(pl_dtype).alias(col + "_" + method)
                else:
                    e = getattr(pl.col(col).dt, method)().cast(pl_dtype).alias(col + "_" + method)

                all_exprs.append(e)
        df = df.with_columns(all_exprs)

        if delete_original_cols:
            df = df.drop(cols)

    return df


def run_pysr_fe(
    df: pl.DataFrame,
    nsamples: int = 100_000,
    target_columns_prefix: str = "target_",
    timeout_mins: int = 5,
    fill_nans: bool = True,
):
    """Fit a PySR symbolic regressor on a sampled polars frame.

    DEPRECATED: use :func:`mlframe.feature_engineering.bruteforce.run_pysr_feature_engineering` instead -- that path adds leakage-free OOF support, preset wiring, reserved-name handling, and the global Julia lock. This legacy single-target entry point is retained for back-compat only.

    The frame is split into features (numeric columns NOT prefixed by ``target_columns_prefix``)
    and targets (columns starting with that prefix). Duplicate sanitised names are disambiguated
    with a numeric suffix so PySR sees a unique feature set.

    When exactly one ``target_<...>`` column is detected the body delegates to
    ``bruteforce.run_pysr_feature_engineering`` so callers get the leakage-free OOF + Julia-lock
    plumbing automatically. Multi-target frames (``targets_df.shape[1] > 1``) still take the
    legacy in-place path because the new entry point is single-output by design; the warning
    surfaces this so users migrate explicitly when they need the multi-output behaviour.
    """
    import warnings

    target_cols = [c for c in df.columns if c.startswith(target_columns_prefix)]
    if len(target_cols) == 1:
        warnings.warn(
            "mlframe.feature_engineering.basic.run_pysr_fe is deprecated; delegating to "
            "bruteforce.run_pysr_feature_engineering. Call that function directly to access "
            "preset wiring + random_state + leakage-free OOF knobs.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .bruteforce import run_pysr_feature_engineering

        drop_columns = [c for c in df.columns if c.startswith(target_columns_prefix) and c != target_cols[0]]
        # nsamples<=0 means "use the full frame" in the legacy semantics; the bruteforce entry
        # caps at sample_size so we pass len(df) as the "no cap" equivalent.
        sample_size = nsamples if nsamples and nsamples > 0 else len(df)
        pysr_params_override = {
            "turbo": True,
            "timeout_in_seconds": timeout_mins * 60,
            "maxsize": 10,
            "niterations": 10,
            "binary_operators": ["+", "*"],
            "unary_operators": ["cos", "exp", "log", "sin", "inv(x) = 1/x"],
            "extra_sympy_mappings": {"inv": lambda x: 1 / x},
            "elementwise_loss": "loss(prediction, target) = abs(prediction - target)",
        }
        return run_pysr_feature_engineering(
            df,
            target_col=target_cols[0],
            drop_columns=drop_columns,
            sample_size=sample_size,
            pysr_params_override=pysr_params_override,
            leakage_free=False,
            verbose=0,
        )

    warnings.warn(
        "mlframe.feature_engineering.basic.run_pysr_fe is deprecated and only retains the multi-target "
        "in-place path because bruteforce.run_pysr_feature_engineering is single-output. Migrate to "
        "per-target loops over bruteforce.run_pysr_feature_engineering for leakage-free OOF + Julia-lock support.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pysr import PySRRegressor

    clean_ram()

    model = PySRRegressor(
        turbo=True,
        timeout_in_seconds=timeout_mins * 60,
        maxsize=10,
        niterations=10,
        binary_operators=["+", "*"],
        unary_operators=["cos", "exp", "log", "sin", "inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = abs(prediction - target)",
    )

    # Two-pass replace could collapse distinct columns onto the same sanitised name
    # (e.g. "a=b" and "a.b" both become "a_b"). Deduplicate by appending an index suffix on collision.
    rename_map: Dict[str, str] = {}
    used: set = set()
    for col in df.columns:
        candidate = col.replace("=", "_").replace(".", "_")
        base = candidate
        i = 1
        while candidate in used:
            candidate = f"{base}__{i}"
            i += 1
        used.add(candidate)
        rename_map[col] = candidate

    # nsamples<=0 means "use the full frame"; nsamples>0 caps the row count.
    tmp_df = df.head(nsamples) if nsamples and nsamples > 0 else df
    expr = cs.numeric() - cs.starts_with(target_columns_prefix)
    if fill_nans:
        expr = expr.fill_null(0).fill_nan(0)

    features_df = tmp_df.select(expr).rename(rename_map)
    targets_df = tmp_df.select(cs.starts_with(target_columns_prefix))
    model.fit(features_df, targets_df)

    del tmp_df, features_df, targets_df
    clean_ram()

    return model
