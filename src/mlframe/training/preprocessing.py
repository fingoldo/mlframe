"""
Data preprocessing functions for mlframe training pipeline.

Handles data loading, cleaning, train/val/test splitting, and artifact saving.
"""

from __future__ import annotations

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
import pandas as pd
import polars as pl
from typing import Union, Optional, Tuple, Any
import os
from os.path import join
from pyutilz.pandaslib import ensure_dataframe_float32_convertability
from pyutilz.system import ensure_dir_exists
from pyutilz.strings import slugify

from .utils import (
    process_infinities,
    remove_constant_columns,
    save_series_or_df,
    log_ram_usage,
)
from .configs import PreprocessingConfig


def _process_special_values(
    df: Union[pl.DataFrame, pd.DataFrame],
    fill_value: float = 0.0,
    verbose: int = 1,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """Normalise inf to NaN on numeric columns and log null / NaN / inf telemetry.

    Nulls and NaNs are intentionally NOT filled here: pre-split null-fill biases the downstream imputer (e.g. SimpleImputer.fit) into computing mean/median over zero-padded rows
    instead of the real non-null distribution. The imputer is fit on train only after the split and learns the unbiased statistic, then transforms val/test consistently.

    inf -> NaN (not ``fill_value``) so the downstream imputer treats it as missing rather than swallowing a sentinel value into the learned statistic.

    ``fill_value`` is kept in the signature for back-compat with callers that still pass it; the value itself is no longer applied here.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        # Single combined diagnostic scan
        try:
            import polars.selectors as cs
        except ImportError:
            return df
        diag = df.select(
            cs.numeric().is_null().sum().name.prefix("nulls_"),
            cs.numeric().is_nan().sum().name.prefix("nans_"),
            cs.numeric().is_infinite().sum().name.prefix("infs_"),
        )
        # Only inf is rewritten -> NaN so the downstream imputer sees a real
        # missing marker. Restrict to float columns: integer columns cannot
        # hold inf, and polars raises (or silently no-ops depending on
        # version) when ``.replace([inf, -inf], nan)`` is targeted at integer
        # dtypes because ``nan`` is not representable as an integer.
        df = df.with_columns(cs.float().replace([float("inf"), float("-inf")], float("nan")))
        if verbose and diag.height > 0:
            row = diag.row(0)
            parts = []
            # ``row`` is a tuple of scalar counts (one per numeric column per
            # kind); tuples have no ``.max()``. Use the builtin ``max`` on
            # the slice, and only when non-empty.
            for kind, vals in [
                ("null", row[: len(row) // 3]),
                ("NaN", row[len(row) // 3 : 2 * len(row) // 3]),
                ("inf", row[2 * len(row) // 3 :]),
            ]:
                if vals and max(vals) > 0:
                    parts.append(f"{kind}={max(vals)}")
            if parts:
                logger.info("Preprocessing: %s", ", ".join(parts))
    else:
        # Pandas: matching telemetry on the same numeric subset. NaN is the pandas null surrogate (np.nan), so ``.isna()`` covers both null and NaN; inf is a
        # separate np.isinf check on the float subset (integers cannot carry inf).
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            if verbose:
                num_view = df[num_cols]
                null_count = int(num_view.isna().sum().sum())
                # _pandas_float_like_columns covers both legacy np.float and pandas nullable
                # extension Float dtypes; the prior select_dtypes(include="floating") silently
                # MISSED pd.Float32Dtype / Float64Dtype columns and inf in them slipped past
                # the diagnostic + the scrub below, then crashed XGB/HGB downstream.
                _float_cols = _pandas_float_like_columns(df)
                if _float_cols:
                    # to_numpy(na_value=nan) so nullable Float arrays survive the np.isinf call.
                    _inf_arr = df[_float_cols].to_numpy(dtype=np.float64, na_value=np.nan)
                    inf_count = int(np.isinf(_inf_arr).sum())
                else:
                    inf_count = 0
                parts = []
                if null_count > 0:
                    parts.append(f"null/NaN={null_count}")
                if inf_count > 0:
                    parts.append(f"inf={inf_count}")
                if parts:
                    logger.info("Preprocessing: %s", ", ".join(parts))
            # Restrict inf -> NaN to floats: integer columns cannot hold inf and pandas .replace on int dtypes would coerce them to float unnecessarily.
            float_cols = _pandas_float_like_columns(df)
            if float_cols:
                df[float_cols] = df[float_cols].replace([float("inf"), float("-inf")], float("nan"))
    return df


# Backward-compat alias - some callers still import the old fused name.
# Kept as a thin pass-through so existing imports don't break; the
# canonical name above drops the misleading "_fused" suffix.
_process_special_values_fused = _process_special_values


def _pandas_float_like_columns(df) -> list:
    """Return the names of every "float-like" column in a pandas DataFrame.

    Includes BOTH the legacy numpy float dtypes (``float32`` / ``float64``,
    selected by ``select_dtypes(include="floating")``) AND the pandas nullable
    extension Float dtypes (``Float32Dtype`` / ``Float64Dtype``), which the
    bare ``"floating"`` selector silently SKIPS because they're
    ``ExtensionDtype`` not legacy ``np.floating``.

    Pre-fix shape (commits before 2026-05-20 resource wave): callers that
    used ``df.select_dtypes(include="floating")`` to identify columns for
    inf-scrubbing or inf-detection missed pandas nullable Float columns;
    inf values passed through silently and crashed XGB / HGB deep in C++
    with no log line pointing back to the unscrubbed column.

    Mirrors polars ``cs.float()`` which already covers both regular and
    extension polars float dtypes uniformly.
    """
    import pandas as _pd
    cols: list = []
    for _col in df.columns:
        _dt = df[_col].dtype
        if _pd.api.types.is_float_dtype(_dt):
            # is_float_dtype returns True for both np.float32/64 AND pd.Float32Dtype/Float64Dtype
            cols.append(_col)
    return cols


def _frame_contains_inf(df) -> bool:
    """Cheap O(numeric_cols * n_rows) scan that returns True iff ``df``
    contains any ``+inf`` / ``-inf`` in a numeric column.

    Used by the ``fix_infinities=False`` path in ``preprocess_dataframe``
    to fail loud (auto-fix + ERROR log) when the user opted out of
    inf-handling but the data actually contains inf — better than an
    opaque XGB / HGB crash deep in C++.
    """
    try:
        if isinstance(df, pl.DataFrame):
            for name, dtype in df.schema.items():
                if not dtype.is_numeric():
                    continue
                if df[name].is_infinite().any():
                    return True
            return False
        # pandas: previously used select_dtypes(include=["floating"]) which silently
        # MISSED pandas nullable Float (Float32Dtype / Float64Dtype) -- inf in those
        # columns would slip past the fix_infinities=False loud-fail check too, and
        # then crash XGB/HGB downstream with no log line. Route through the
        # _pandas_float_like_columns helper that covers both legacy and extension floats.
        try:
            _float_cols = _pandas_float_like_columns(df)
        except (AttributeError, TypeError):
            return False
        if not _float_cols:
            return False
        num = df[_float_cols]
        if num.shape[1] == 0:
            return False
        try:
            # na_value=nan so pandas nullable Float dtypes don't fall through to object dtype.
            return bool(np.isinf(num.to_numpy(dtype=np.float64, na_value=np.nan)).any())
        except (TypeError, ValueError) as _e_inner:
            # Specific failure here (e.g. nullable Float64Dtype that
            # to_numpy can't coerce): WARN-log + treat as "unknown"
            # rather than False, then let the caller's fix_infinities
            # branch decide. Returning False silently here would tell
            # the caller "no infs detected" and XGB/HGB would crash
            # later with no upstream signal.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "_has_any_infinity: numpy conversion failed (%s); cannot "
                "confirm absence of inf - returning True to force the "
                "fix_infinities path (safer than silently passing infs "
                "downstream to XGB/HGB).", _e_inner,
            )
            return True
    except Exception as _e_outer:
        # Outer failure (e.g. df doesn't support .schema or _pandas_float_like_columns
        # helper itself broke). Pre-fix returned False silently which let infs reach
        # the booster. Same conservative branch: assume infs MAY be present.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "_has_any_infinity: detection failed entirely (%s); returning True "
            "to force the fix_infinities path. The original detector exists "
            "specifically to flag this case.", _e_outer,
        )
        return True


def load_and_prepare_dataframe(
    df: Union[pl.DataFrame, str],
    config: PreprocessingConfig,
    verbose: int = 1,
) -> pl.DataFrame:
    """
    Load and prepare dataframe for training (Polars only).

    Args:
        df: Polars DataFrame or path to parquet file
        config: Preprocessing configuration
        verbose: Verbosity level

    Returns:
        Polars DataFrame

    Notes:
        - Only supports Polars (for efficiency)
        - Column dropping happens AFTER features_and_targets_extractor.transform() in core.py
          (columns might be needed by features_and_targets_extractor or created by it)
        - If both n_rows and tail are set, tail is applied AFTER n_rows. So n_rows=1000
          with tail=100 gives the last 100 of the first 1000 rows, not the last 100 of the file.
    """
    # Load from file if path provided
    if isinstance(df, str):
        if verbose:
            logger.info("Loading dataframe from %s with Polars...", df)

        if not df.lower().endswith(".parquet"):
            raise ValueError(f"Only parquet format supported, got: {df}")

        # Build efficient loading parameters
        load_params = {"parallel": "columns"}

        # Use n_rows at load time for efficiency
        if config.n_rows:
            load_params["n_rows"] = config.n_rows
            if verbose:
                logger.info("Loading first %s rows...", config.n_rows)

        # Use columns at load time for efficiency
        if config.columns:
            load_params["columns"] = config.columns
            if verbose:
                logger.info(f"Loading {len(config.columns)} columns...")

        # Use read_parquet if columns/n_rows specified (scan_parquet has a narrower kwarg surface
        # and does not accept `parallel="columns"` or `columns=`). Otherwise scan lazily and let
        # Polars collect once downstream - keeps memory low for wide files.
        if config.columns or config.n_rows:
            df = pl.read_parquet(df, **load_params)
        else:
            # scan_parquet rejects `parallel="columns"` (only valid on eager read_parquet).
            df = pl.scan_parquet(df)

    # Apply tail if specified (after loading). For a LazyFrame produced by scan_parquet,
    # collect via the streaming engine so the OS only ever needs (tail_size + per-batch buffer)
    # rows in memory rather than the whole file (fix for audit B-P0-7: scan_parquet -> tail ->
    # plain .collect() can OOM on a 100GB file even when tail=10k because slice-pushdown does
    # not always reach the parquet reader; streaming guarantees a bounded working set).
    if config.tail:
        if verbose:
            logger.info("Taking last %s rows...", config.tail)
        df = df.tail(config.tail)

    if isinstance(df, pl.LazyFrame):
        try:
            df = df.collect(streaming=True)
        except TypeError:
            # polars version without the streaming kwarg falls back to default collect.
            df = df.collect()

    if verbose:
        log_ram_usage()

    return df


def preprocess_dataframe(
    df: Union[pl.DataFrame, pd.DataFrame],
    config: PreprocessingConfig,
    verbose: int = 1,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Preprocess dataframe: handle nulls, NaNs, infinities, constants, dtypes.

    Args:
        df: Input DataFrame
        config: Preprocessing configuration
        verbose: Verbosity level

    Returns:
        Preprocessed DataFrame
    """
    original_shape = df.shape

    # Normalise pandas StringDtype -> object so downstream dtype checks
    # (`dtype == object`, `is_object_dtype`) keep working. pandas 2.2+ may
    # back string columns with pyarrow / nullable StringDtype by default;
    # mlframe code paths that gate on object-dtype (auto-detect feature
    # types, baseline diagnostics, LGB/XGB cat conversion, pre-screen)
    # then silently skip those columns and the model fits crash with
    # ``pandas dtypes must be int, float or bool, got cat_a: str``
    # (observed 2026-05-20 on S: in test_each_ensemble_method_writes_its_own_perfplot).
    # Treat StringDtype as legacy object; the explicit cat-handling path
    # converts to ``category`` downstream when appropriate.
    if isinstance(df, pd.DataFrame):
        # Pandas allows duplicate column names; ``df[c]`` for a duplicate-named
        # ``c`` returns a DataFrame (not a Series) which has no ``.dtype``.
        # Walk dtypes positionally instead so the column-by-name lookup never
        # fires on duplicates - the dedicated dup-column raise downstream
        # handles surfacing the real error to the caller.
        _string_cols = [c for c, _dt in zip(df.columns, df.dtypes) if isinstance(_dt, pd.StringDtype)]
        if _string_cols:
            # Shallow copy: only StringDtype columns are recast below; deep-copying a 100+ GB frame to normalise a few columns OOMs. ``deep=False`` shares untouched buffers, caller frame unmutated.
            df = df.copy(deep=False)
            for _c in _string_cols:
                df[_c] = df[_c].astype(object)
            if verbose:
                logger.info(
                    "Normalised %d pandas StringDtype column(s) -> object: %s",
                    len(_string_cols), _string_cols,
                )

    # iter291 (2026-05-26): batched-scan fastpath for polars frames.
    # ``remove_constant_columns`` + ``process_infinities`` legacy chain
    # walks the frame THREE times (constant-numeric, constant-nonnumeric,
    # infinite-count) via three sequential ``df.select(...)`` calls. On
    # wide post-pipeline polars frames (200k x 1000+ cols after polynomial
    # + RFF) c0140 iter291 attributed 60.7s cumulative to those three
    # passes. Bundle the three aggregations into one ``df.select(...)``
    # so polars' query planner can fuse the data sweep; ~1.15x on
    # synthetic 205-col frames and proportionally more on real wide
    # post-pipeline shapes where per-scan constant overhead dominates.
    #
    # Bypassed when ``MLFRAME_DISABLE_BATCHED_PREPROCESS_SCAN=1`` so the
    # original three-pass path is restored for forensic A/B benchmarks.
    _do_remove_const = bool(getattr(config, "remove_constant_columns", True))
    _do_fix_inf = bool(config.fix_infinities) and config.fillna_value is None
    _use_batched = isinstance(df, pl.DataFrame) and (_do_remove_const or _do_fix_inf) and os.environ.get("MLFRAME_DISABLE_BATCHED_PREPROCESS_SCAN") != "1"
    if _use_batched:
        from ._nan_processing import batch_scan_constants_and_inf_polars
        import polars.selectors as _cs
        scan = batch_scan_constants_and_inf_polars(
            df,
            detect_constant_numeric=_do_remove_const,
            detect_constant_nonnumeric=_do_remove_const,
            detect_inf=_do_fix_inf,
        )
        if _do_remove_const:
            _drop = scan["constant_numeric"] + scan["constant_nonnumeric"]
            if _drop:
                if verbose:
                    logger.info(
                        "Removing %d constant column(s) (numeric: %d, non-numeric: %d): %s",
                        len(_drop), len(scan["constant_numeric"]),
                        len(scan["constant_nonnumeric"]), _drop,
                    )
                df = df.drop(_drop)
        if _do_fix_inf and scan["inf_counts"]:
            if verbose:
                logger.info(
                    "Replacing infinities in %d numeric column(s) (total %d rows): %s",
                    len(scan["inf_counts"]), sum(scan["inf_counts"].values()),
                    sorted(scan["inf_counts"]),
                )
            df = df.with_columns(_cs.float().replace([float("inf"), float("-inf")], 0.0))
        # Float32 cast is the only remaining work; the fillna_value=None
        # branch is unreachable in the batched fastpath since _do_fix_inf
        # gates on fillna_value is None.
    else:
        # Legacy 3-pass path (pandas, or batched fastpath disabled).
        # Remove constant columns (2026-04-21: gated on config flag; default True).
        if _do_remove_const:
            df = remove_constant_columns(df, verbose=verbose)

    # Ensure float32 dtypes if requested (works for both pandas and Polars)
    if config.ensure_float32_dtypes:
        df = ensure_dataframe_float32_convertability(df)

    # Inf is normalised to NaN here so the post-split imputer (SimpleImputer / polars-ds Blueprint) treats it as missing. Pre-split null-fill was removed: filling
    # before the split biases the imputer's learned mean/median toward the sentinel value (e.g. 70 instead of true 100 when 30 percent of rows are 0-padded).
    # ``fillna_value`` is retained as a config flag so existing call sites still hit this branch; the value itself is no longer applied pre-split.
    if config.fillna_value is not None:
        df = _process_special_values(df, fill_value=config.fillna_value, verbose=verbose)
    elif not _use_batched and config.fix_infinities:
        df = process_infinities(df, fill_value=0.0, verbose=verbose)
    elif not _use_batched and _frame_contains_inf(df):
        logger.error(
            "fix_infinities=False but data contains np.inf in numeric "
            "columns. Auto-fixing to 0.0 to avoid an opaque XGB / HGB / "
            "sklearn crash later. Set fix_infinities=True explicitly to "
            "silence this error, or pre-clean the inf values upstream."
        )
        df = process_infinities(df, fill_value=0.0, verbose=verbose)

    if verbose:
        logger.info("Preprocessing: %s -> %s", original_shape, df.shape)
        log_ram_usage()

    return df


def _positional_take(obj, idx):
    """Positional row selection for pandas / polars / numpy.

    ``train_idx``/``val_idx``/``test_idx`` are POSITIONAL integer arrays (the
    splitter and ``create_split_dataframes`` consume them via ``.iloc`` / polars
    ``[]``). A bare ``pandas_series[idx]`` is LABEL-based, so on a non-RangeIndex
    frame it either raises or silently selects the WRONG rows -- saving split
    artifacts that are misaligned with the rows the model trained on. Use
    ``.iloc`` for pandas; polars/numpy ``[]`` is already positional.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.iloc[idx]
    return obj[idx]


def save_split_artifacts(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    timestamps: Optional[Union[pd.Series, pl.Series]],
    group_ids_raw: Optional[Union[np.ndarray, pd.Series, pl.Series]],
    artifacts: Optional[Any],
    data_dir: Optional[str],
    models_dir: str,
    target_name: str,
    model_name: str,
    compression: str = "zstd",
):
    """
    Save split artifacts (timestamps, group_ids, artifacts) for each split.

    Args:
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        timestamps: Timestamp series
        group_ids_raw: Group ID series/array
        artifacts: Additional artifacts from features_and_targets_extractor
        data_dir: Base data directory
        models_dir: Models subdirectory
        target_name: Target name (for directory structure)
        model_name: Model name (for directory structure)
        compression: Compression algorithm
    """
    if data_dir is not None and models_dir:
        # Hoist invariant path join out of the per-split loop.
        split_dir = join(data_dir, models_dir, slugify(target_name), slugify(model_name))
        ensure_dir_exists(split_dir)

        # Single listdir instead of per-file exists() — avoids N×17ms os.stat on Windows
        # where antivirus scanning makes each stat call ~17ms.
        try:
            _existing = set(os.listdir(split_dir))
        except OSError:
            _existing = set()

        for idx, idx_name in zip([train_idx, val_idx, test_idx], "train val test".split()):
            if idx is None:
                continue
            if timestamps is not None and len(timestamps) > 0:
                ts_fname = f"{idx_name}_timestamps.parquet"
                if ts_fname not in _existing:
                    save_series_or_df(_positional_take(timestamps, idx), join(split_dir, ts_fname), compression, name="ts")
            if group_ids_raw is not None and len(group_ids_raw) > 0:
                gid_fname = f"{idx_name}_group_ids_raw.parquet"
                if gid_fname not in _existing:
                    save_series_or_df(_positional_take(group_ids_raw, idx), join(split_dir, gid_fname), compression)
            if artifacts is not None and len(artifacts) > 0:
                if isinstance(artifacts, dict):
                    # Per-key artifacts: write one parquet file per dict entry.
                    for art_key, art_val in artifacts.items():
                        if art_val is None:
                            continue
                        art_fname = f"{idx_name}_artifacts_{slugify(str(art_key))}.parquet"
                        if art_fname not in _existing:
                            save_series_or_df(_positional_take(art_val, idx), join(split_dir, art_fname), compression)
                else:
                    art_fname = f"{idx_name}_artifacts.parquet"
                    if art_fname not in _existing:
                        save_series_or_df(_positional_take(artifacts, idx), join(split_dir, art_fname), compression)


def create_split_dataframes(
    df: Union[pd.DataFrame, pl.DataFrame],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame]]:
    """
    Create train, val, test dataframes from indices.

    Args:
        df: Original DataFrame
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        train_df = df[train_idx]
        val_df = df[val_idx] if len(val_idx) > 0 else pl.DataFrame()
        test_df = df[test_idx] if len(test_idx) > 0 else pl.DataFrame()
    else:
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx] if len(val_idx) > 0 else pd.DataFrame()
        test_df = df.iloc[test_idx] if len(test_idx) > 0 else pd.DataFrame()

    return train_df, val_df, test_df


__all__ = [
    "load_and_prepare_dataframe",
    "preprocess_dataframe",
    "save_split_artifacts",
    "create_split_dataframes",
]
