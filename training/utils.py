"""
Utility functions for mlframe training pipeline.

Functions for RAM management, file I/O, dataframe conversions, and data cleaning.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging
import sys

logger = logging.getLogger(__name__)


def _caller_logger() -> logging.Logger:
    """Return the logger bound to the module that called the public helper
    which in turn called us. Used so progress lines like "Done. RAM usage:"
    or the "PHASE N" banner are attributed to the caller's module (e.g.
    ``mlframe.training.core``) instead of this utils module — matches what
    a reader expects when scanning log origins.
    """
    try:
        # Stack: [_caller_logger] -> [public helper] -> [real caller]
        frame = sys._getframe(2)
        return logging.getLogger(frame.f_globals.get("__name__", __name__))
    except Exception:
        return logger

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import os
from textwrap import shorten
from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import polars.selectors as cs
from pyutilz.system import clean_ram, get_own_memory_usage


def log_ram_usage() -> None:
    """Log current RAM usage, attributed to the caller's module."""
    _caller_logger().info(f"Done. RAM usage: {get_own_memory_usage():.1f}GB.")


# Adaptive clean_ram: skip gc.collect + trim when RSS hasn't grown meaningfully
# since the last clean. On small-DF runs (<50MB), gc.collect alone costs ~0.4s/call;
# 10 unconditional calls is ~44% of a 10s training. Baseline is refreshed after
# every real clean, so growth is measured from the most recent state.
_MAYBE_CLEAN_BASELINE_MB: float = 0.0
_MAYBE_CLEAN_MIN_GROWTH_MB: float = 500.0


def maybe_clean_ram_adaptive() -> None:
    """Call pyutilz.clean_ram only when process RSS has grown by
    ``_MAYBE_CLEAN_MIN_GROWTH_MB`` since the previous clean. Cheap
    short-circuit replacement for bare ``clean_ram()`` on hot training
    paths where small-DF runs don't justify a 0.4s gc.collect per call.
    """
    global _MAYBE_CLEAN_BASELINE_MB
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024**2
    except Exception:
        clean_ram()
        return
    if _MAYBE_CLEAN_BASELINE_MB == 0.0:
        _MAYBE_CLEAN_BASELINE_MB = rss_mb
        return
    if rss_mb - _MAYBE_CLEAN_BASELINE_MB > _MAYBE_CLEAN_MIN_GROWTH_MB:
        clean_ram()
        try:
            _MAYBE_CLEAN_BASELINE_MB = psutil.Process().memory_info().rss / 1024**2
        except Exception:
            _MAYBE_CLEAN_BASELINE_MB = rss_mb


def clean_ram_and_gpu(verbose: bool = False) -> None:
    """
    Clean both CPU RAM and GPU memory.

    Combines pyutilz.clean_ram() with GPU memory cleanup.
    Call this after model training to free memory before training next model.

    Args:
        verbose: If True, log memory stats after cleanup
    """
    import gc

    # Clean CPU RAM first
    clean_ram()

    # Clean GPU memory if PyTorch CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            # Synchronize all CUDA streams before cleanup
            torch.cuda.synchronize()
            # Empty the CUDA memory cache
            torch.cuda.empty_cache()
            # Force garbage collection again after GPU cleanup
            gc.collect()

            if verbose:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except ImportError:
        pass  # PyTorch not installed


def estimate_df_size_mb(df) -> float:
    """Estimated in-memory size of a Polars/pandas DataFrame in MB.

    Returns `inf` for unsupported types so downstream OOM-protection thresholds
    trip correctly (Arrow/Modin/Dask inputs otherwise silently lose `clean_ram`
    heuristic's size-proportional growth check).
    """
    if isinstance(df, pl.DataFrame):
        return float(df.estimated_size("mb"))
    if isinstance(df, pd.DataFrame):
        return float(df.memory_usage(deep=True).sum() / 1024**2)
    return float("inf")


def get_process_rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def should_clean_ram(baseline_rss_mb: float, df_size_mb: float, min_growth_mb: float = 500.0) -> bool:
    """True iff a clean_ram call (~0.6s) is likely justified.

    Triggers when either:
      - RSS grew beyond baseline by max(min_growth_mb, 30% of DF size) — accumulated
        temp state worth collecting; OR
      - free system RAM < 2x DF size — OOM risk, gc may release Arrow buffers.
    """
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024**2
        free_mb = psutil.virtual_memory().available / 1024**2
    except Exception as e:
        logger.debug("should_clean_ram: RAM measurement failed, falling back to clean", exc_info=e)
        return True  # can't measure → fall back to cleaning
    growth = rss_mb - baseline_rss_mb
    return (growth > max(min_growth_mb, 0.3 * df_size_mb)) or (free_mb < 2 * df_size_mb)


def maybe_clean_ram_and_gpu(
    baseline_rss_mb: float,
    df_size_mb: float,
    verbose: bool = False,
    reason: str = "",
) -> float:
    """Call clean_ram_and_gpu only when RAM metrics indicate it's worthwhile.

    On small DFs this avoids 0.6s of pure overhead per call; on large production
    DFs (or when the process is growing) it still fires at every site.

    Returns the (possibly refreshed) baseline RSS in MB. After a fire, baseline
    is re-captured so subsequent `growth = rss - baseline` checks are not
    monotonically inflated by already-cleaned state. Callers should assign
    the return back to their local baseline variable.
    """
    if should_clean_ram(baseline_rss_mb, df_size_mb):
        clean_ram_and_gpu(verbose=verbose)
        if verbose:
            logger.info(f"  clean_ram fired ({reason})" if reason else "  clean_ram fired")
        return get_process_rss_mb()
    return baseline_rss_mb


def filter_existing(df, cols) -> list:
    """Return only the column names from `cols` that exist in `df.columns`.

    Preserves input order. Accepts any iterable of column names. When `df`
    has no `columns` attribute (e.g. numpy ndarray), returns an empty list
    so callers can safely chain without pre-checks.
    """
    columns = getattr(df, "columns", None)
    if columns is None:
        return []
    existing = set(columns)
    return [c for c in cols if c in existing]


def log_phase(msg: str, n: int = 80) -> None:
    """Log a single separator line followed by the phase message.

    Width 80 reads comfortably on terminals and in notebook cells (was 160 —
    wrapped horizontally). Only one separator per call: consecutive
    ``log_phase`` calls render as::

        ---
        First phase msg
        ---
        Second phase msg

    avoiding the "two adjacent dash lines" banner noise the previous
    top+bottom layout produced.
    """
    lg = _caller_logger()
    lg.info("-" * n)
    lg.info(msg)


def drop_columns_from_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame],
    additional_columns_to_drop: Optional[list] = None,
    config_drop_columns: Optional[list] = None,
    verbose: int = 1,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Drop columns from dataframe.

    Args:
        df: DataFrame to drop columns from
        additional_columns_to_drop: Additional columns to drop (e.g., from preprocessor)
        config_drop_columns: Columns from config to drop
        verbose: Verbosity level

    Returns:
        DataFrame with columns dropped
    """
    if not additional_columns_to_drop and not config_drop_columns:
        return df

    all_cols_to_drop = []
    if additional_columns_to_drop:
        all_cols_to_drop.extend(additional_columns_to_drop)
    if config_drop_columns:
        all_cols_to_drop.extend(config_drop_columns)

    # Remove duplicates
    all_cols_to_drop = set(all_cols_to_drop)

    if not all_cols_to_drop:
        return df

    if verbose:
        logger.info(f"Dropping {len(all_cols_to_drop)} column(s): {shorten(','.join(all_cols_to_drop),250)}...")

    if isinstance(df, pl.DataFrame):
        df = df.drop(all_cols_to_drop, strict=False)
    else:
        existing_cols = all_cols_to_drop.intersection(set(df.columns))
        if existing_cols:
            df = df.drop(columns=existing_cols)

    if verbose:
        log_ram_usage()

    return df


# NOTE: save_mlframe_model and load_mlframe_model are in io.py
# Use: from .io import save_mlframe_model, load_mlframe_model


_NESTED_DTYPE_WARN_SEEN: set = set()
"""Module-level dedup cache for ``get_pandas_view_of_polars_df``'s nested-
dtypes WARN. Keeps the bridge quiet on repeated calls with the same
embedding-column schema while still surfacing novel shapes. Cleared
implicitly on process exit; tests that need fresh state can clear it
explicitly via ``_NESTED_DTYPE_WARN_SEEN.clear()``."""


def _dtype_family(dtype_str: str) -> str:
    """Map a canonical dtype string to a coarse family token.

    Fix 8 load-time diff uses this to distinguish BENIGN width drift
    (float32 <-> float64, int32 <-> int64) from FAMILY mismatches
    (string -> numeric, numeric -> categorical) that will actually
    break inference. String values are produced by ``_canonical_dtype_str``
    and pandas ``str(dtype)``.
    """
    s = str(dtype_str).lower()
    if s in ("string", "str", "object", "utf8", "categorical") or s.startswith("enum") or s == "category":
        return "string-or-cat"
    if any(tok in s for tok in ("float", "decimal", "double")):
        return "float"
    if any(tok in s for tok in ("int", "uint")):
        return "int"
    if "bool" in s:
        return "bool"
    if "date" in s or "time" in s:
        return "datetime"
    if s.startswith("list"):
        return "list"  # embedding-family
    return s


def _canonical_dtype_str(dtype) -> str:
    """Canonicalise a Polars or pandas dtype into a stable string form for
    hashing. Aliases collapse (pl.Utf8 / pl.String -> 'String'), and pl.Enum
    records both the token and the sorted category tuple so category drift
    is detectable (val-set with a new category yields a different hash).

    Intentionally strict: two runs on the same column with different
    category palettes (e.g. union vs partial) produce different hashes —
    that's the correct signal for Fix 8's fingerprint, since CatBoost's
    internal dict ordering depends on the category list.
    """
    s = str(dtype)
    # pl.String is an alias for pl.Utf8 as of polars 1.x; collapse.
    if s in ("Utf8", "String"):
        return "String"
    # pl.Enum(categories=['a','b','c']) — include categories in canonical form
    # so val-drift with new categories invalidates cached models.
    if s.startswith("Enum("):
        try:
            cats = list(dtype.categories) if hasattr(dtype, "categories") else []
            return "Enum[" + ",".join(sorted(str(c) for c in cats)) + "]"
        except Exception:
            return s
    return s


def compute_model_input_fingerprint(
    df_at_fit,
    cat_features: Optional[list] = None,
    text_features: Optional[list] = None,
    embedding_features: Optional[list] = None,
) -> "tuple[str, list]":
    """Compute a 10-char SHA256 fingerprint of a model's fit-time input
    schema (column names + canonical dtypes + roles).

    Returns:
        (schema_hash, input_schema) where input_schema is the
        order-stable list of ``{"name": str, "dtype": str, "role":
        cat|text|embedding|numeric}`` records used to build the hash.

    Rationale (Fix 8): hashes behaviour — the realised data layout the
    model saw at fit time — not the config flags that produced it. Two
    runs with different config flags (e.g. ``use_text_features=True``
    vs ``False`` for LGB) that yield the same final schema at fit time
    share the same hash, which is the right caching semantics. Changes
    that DO affect the model's input (new column, different dtype,
    dtype-alias swap, role promotion) produce a different hash so the
    cached file isn't silently overwritten.

    Canonical JSON via ``json.dumps(..., sort_keys=True)`` — required by
    the user's memory rule ``feedback_json_hash_sort_keys`` so the hash
    is deterministic across Python builds with different dict orderings.
    """
    import hashlib
    import json as _json

    if df_at_fit is None:
        return "__nodf_____", []

    cat_set = set(cat_features or [])
    text_set = set(text_features or [])
    emb_set = set(embedding_features or [])

    schema = []
    for col in sorted(df_at_fit.columns):
        try:
            if isinstance(df_at_fit, pl.DataFrame):
                dt = df_at_fit.schema[col]
            else:
                dt = df_at_fit[col].dtype
        except Exception:
            dt = "?"
        if col in cat_set:
            role = "cat"
        elif col in text_set:
            role = "text"
        elif col in emb_set:
            role = "embedding"
        else:
            role = "numeric"
        schema.append({"name": col, "dtype": _canonical_dtype_str(dt), "role": role})

    canonical = _json.dumps(schema, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return digest, schema


def get_pandas_view_of_polars_df(
    df: pl.DataFrame,
    self_destruct: bool = False,
    log_threshold_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Return a zero-copy (Arrow-backed) pandas DataFrame view of a Polars DataFrame.

    Args:
        df: Polars DataFrame

    Returns:
        Zero-copy pandas DataFrame view

    Notes:
        - Numeric, boolean, string columns: zero-copy Arrow view.
        - Categorical (dictionary) columns: preserved as ``pd.Categorical``
          (integer codes + categories dict). Polars emits dict arrays with
          uint32 indices, which pyarrow's ``to_pandas`` refuses; we rebuild
          each dict column with int32 indices so the conversion produces
          a proper ``pd.Categorical`` rather than raising.
        - Earlier versions cast dict→string here. That was ~37% slower
          end-to-end on CatBoost (string hashing in fit + predict) and
          OOMed on 450k+ rows with many Categorical columns. Benchmarked
          2026-04-17 (see ``bench_polars_to_pandas.py``).

    Tried but reverted (2026-04-19):
        A ``shared_dict_cache`` parameter was added to share the ``categories``
        pyarrow Array across sliced train/val/test calls, under the assumption
        that slicing a Polars DataFrame preserves the source frame's full
        categorical palette. The benchmark proved otherwise: on every slice
        Polars re-trims the Categorical dictionary to exactly the values
        present in that slice, so train/val/test each carry **different**
        dictionaries (different sets of unique values, different order,
        different length). The cache's safety check correctly bypassed every
        cross-call reuse — net speedup was zero. See the 2026-04-19 CHANGELOG
        entry for the full investigation. If production-grade dict sharing is
        ever needed, the right primitive is ``pl.Enum`` on the source column
        (fixed domain preserved across slices), not a post-hoc Arrow-level
        cache.
    """
    if not isinstance(df, (pl.DataFrame, pl.Series)):
        raise TypeError(f"Input must be a Polars DataFrame or Series, got {type(df).__name__}")

    # Capture timing + RAM around the conversion. Long conversions (multi-GB
    # frames) on prod were silent black boxes — operators couldn't tell whether
    # the 35-min stall was here or downstream. Log when the conversion crosses
    # log_threshold_seconds so small bridge calls don't spam the log.
    import time as _time
    _t0 = _time.perf_counter()
    _rss_before_gb = get_own_memory_usage()
    _shape_str = f"{df.shape[0]:_}×{df.shape[1]}" if isinstance(df, pl.DataFrame) else f"{len(df):_}"

    # Diagnostic: warn on nested Polars types that pyarrow's default
    # ``to_pandas()`` materializes as ``object`` dtype with Python list/
    # dict elements (2026-04-19 probe finding). Downstream CatBoost's
    # embedding_features fastpath expects numeric vectors, not object
    # dtype with list elements, and fails at model.fit() with an opaque
    # "expected numeric" error. Warn here so the operator traces back
    # to the bridge step rather than CatBoost internals. We don't
    # raise or auto-cast — the bridge is a general-purpose helper and
    # other callers (logging, post-hoc analysis) legitimately want
    # list-typed columns pass-through.
    #
    # Noise-dedupe (2026-04-19 round-11): the bridge is called many
    # times per training run (train / val / test × per-model), and a
    # frame with embedding features would emit the same WARN every call.
    # Fire at most once per unique (column-name, dtype-str) tuple set
    # observed in the process — repeated identical schemas stay quiet.
    if isinstance(df, pl.DataFrame):
        nested_cols = []
        for name, dt in df.schema.items():
            type_str = str(dt)
            if any(
                marker in type_str for marker in ("List", "Struct", "Array", "Object")
            ):
                nested_cols.append((name, type_str))
        if nested_cols:
            key = tuple(nested_cols)
            if key not in _NESTED_DTYPE_WARN_SEEN:
                _NESTED_DTYPE_WARN_SEEN.add(key)
                logger.warning(
                    "get_pandas_view_of_polars_df: %d column(s) have nested "
                    "Polars dtypes that pyarrow materializes as pandas object "
                    "dtype with Python list/dict elements: %s. Downstream "
                    "numeric consumers (CatBoost embedding_features fastpath, "
                    "sklearn estimators) may reject these with opaque errors. "
                    "If these columns are embedding_features, keep them as "
                    "pl.List in the Polars fastpath; if they need to hit the "
                    "pandas path, pre-cast to fixed-width numpy arrays. "
                    "(This warning fires at most once per unique schema.)",
                    len(nested_cols), nested_cols,
                )

    tbl = df.to_arrow()

    # Note: short-circuit on "no dictionary columns" was benchmarked 2026-04-14 and
    # delivered only 1.16x on pure-numeric workloads (below 1.2x threshold), so the
    # per-column scan is retained for code uniformity.
    fixed_cols = []
    for col in tbl.columns:
        if pa.types.is_dictionary(col.type):
            chunks = []
            for chunk in col.chunks:
                indices_i32 = pa.compute.cast(chunk.indices, pa.int32())
                chunks.append(
                    pa.DictionaryArray.from_arrays(indices_i32, chunk.dictionary)
                )
            col = pa.chunked_array(chunks)
        fixed_cols.append(col)

    tbl_fixed = pa.table(fixed_cols, names=tbl.column_names)

    # Capture which Arrow columns were bool BEFORE to_pandas(). Nullable Boolean
    # columns (those with any null) materialize as pandas ``object`` dtype with
    # Python True/False/None elements — verified empirically 2026-04-23 against
    # pyarrow 21 on Polars 1.37. This breaks LightGBM's sklearn wrapper, which
    # refuses ``object`` dtypes (``ValueError: pandas dtypes must be int, float
    # or bool``). Non-null Boolean stays as numpy ``bool``. We post-process only
    # the object-materialized ones below.
    _arrow_bool_cols = [
        name for name, col in zip(tbl_fixed.column_names, tbl_fixed.columns)
        if pa.types.is_boolean(col.type)
    ]

    # Use numpy-backed pandas (types_mapper=None) for broad model compatibility.
    # LightGBM (and some other sklearn-family models) reject pandas columns with
    # ArrowDtype (e.g. 'float[pyarrow]').
    #
    # CRITICAL: to_pandas() defaults are NOT zero-copy — they CONSOLIDATE memory
    # blocks across columns of the same dtype, forcing a full copy of every
    # numeric buffer into a fresh numpy array. On 9M × 70 numeric columns this
    # was ~35 minutes of pure memcpy + GIL churn in production.
    #   * split_blocks=True  → no consolidation; each column keeps its Arrow
    #     buffer, so numeric/bool dtypes become np.ndarray views instead of
    #     copies. Categorical still allocates because pandas needs its own
    #     codes representation, but that's the only unavoidable copy.
    #   * use_threads=True   → parallel column materialization (default but
    #     stated explicitly so the contract is visible).
    #   * self_destruct=False (default) → SAFE. Tested 2026-04-22: enabling
    #     self_destruct=True caused a native crash inside pytest during the
    #     integration suite — pyarrow's "EXPERIMENTAL" warning was no joke.
    #     The 26% extra speedup is not worth the segfault risk; opt-in only.
    # Bench (2026-04-22, 7.3M × 118 with 18 dict cols):
    #   default                          30.06s   (1×)
    #   use_threads only                  2.10s   (14×)
    #   +split_blocks                     0.95s   (32×)  ← default
    #   +self_destruct                    0.70s   (43×)  ← opt-in, may crash
    pandas_df = tbl_fixed.to_pandas(
        use_threads=True,
        split_blocks=True,
        self_destruct=self_destruct,
    )

    # Coerce object-materialized nullable Boolean → pandas Int8 with ``pd.NA``.
    # Verified 2026-04-23 that Int8/``pd.NA`` is accepted by all three tree
    # backends (LightGBM 4.6, XGBoost 3.2, CatBoost 1.2.10); pandas nullable
    # ``boolean`` is rejected by CatBoost ("Cannot convert <NA> to float") so we
    # avoid it. Non-null Boolean already came out as numpy ``bool`` and is left
    # alone. Object-elsewhere (e.g. ``pl.List`` / ``pl.Struct``) is untouched —
    # the nested-dtype warning above still fires for those.
    _coerced_bool_cols = []
    for name in _arrow_bool_cols:
        if pandas_df[name].dtype == object:
            pandas_df[name] = pandas_df[name].astype("boolean").astype("Int8")
            _coerced_bool_cols.append(name)
    if _coerced_bool_cols and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "get_pandas_view_of_polars_df: coerced %d nullable-bool object "
            "column(s) to Int8: %s",
            len(_coerced_bool_cols), _coerced_bool_cols,
        )

    _elapsed = _time.perf_counter() - _t0
    if _elapsed >= log_threshold_seconds:
        _rss_after_gb = get_own_memory_usage()
        logger.info(
            "get_pandas_view_of_polars_df %s: %.2fs, RAM %.2f→%.2f GB (Δ%+.2f), "
            "self_destruct=%s",
            _shape_str, _elapsed, _rss_before_gb, _rss_after_gb,
            _rss_after_gb - _rss_before_gb, self_destruct,
        )

    return pandas_df


def save_series_or_df(
    obj: Union[pd.Series, pd.DataFrame, pl.Series, pl.DataFrame],
    file: str,
    compression: str = "zstd",
    name: Optional[str] = None,
) -> None:
    """
    Save a pandas/polars Series or DataFrame to parquet.

    Args:
        obj: Series or DataFrame to save
        file: Output file path
        compression: Compression algorithm (default: zstd)
        name: Name for Series (if converting to DataFrame)

    Raises:
        FileNotFoundError: If the parent directory does not exist.
    """
    parent_dir = os.path.dirname(file)
    if parent_dir and not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

    if isinstance(obj, (pd.Series, pl.Series)):
        if name:
            obj = obj.to_frame(name=name)
        else:
            obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(file, compression=compression)
    elif isinstance(obj, pl.DataFrame):
        obj.write_parquet(file, compression=compression)


def _process_special_values(
    df: Union[pl.DataFrame, pd.DataFrame],
    expr_func: Optional[Callable[[], pl.Expr]] = None,
    fill_func_name: Optional[str] = None,
    kind: str = "",
    fill_value: Optional[float] = None,
    drop_columns: bool = False,
    verbose: int = 1,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Generic handler for NaNs, nulls, infinities, or constant columns in numeric columns.

    Args:
        df: Input DataFrame (Polars or pandas)
        expr_func: Function returning Polars expression for detecting issues
        fill_func_name: Method name for filling values (fill_nan, fill_null, replace)
        kind: Description of issue type (for logging)
        fill_value: Value to fill with
        drop_columns: Whether to drop problematic columns
        verbose: Verbosity level

    Returns:
        Cleaned DataFrame
    """
    is_polars = isinstance(df, pl.DataFrame)

    if verbose:
        logger.info(f"{'Identifying' if drop_columns else 'Counting'} {kind}{'s' if not kind.endswith('columns') else ''}...")

    if is_polars:
        # Polars: use provided expr_func
        qos_df = df.select(expr_func())

        if drop_columns:
            # For constant detection, we get boolean indicators
            if len(qos_df) > 0:
                constant_cols = [col for col, is_const in zip(qos_df.columns, qos_df.row(0)) if is_const]
            else:
                constant_cols = []
            errors_df = pl.DataFrame({"column": constant_cols, "nerrors": [1] * len(constant_cols)}, schema={"column": pl.String, "nerrors": pl.Int64})
        else:
            if len(qos_df) > 0:
                errors_df = pl.DataFrame({"column": qos_df.columns, "nerrors": qos_df.row(0)}).filter(pl.col("nerrors") > 0).sort("nerrors", descending=True)
            else:
                errors_df = pl.DataFrame({"column": [], "nerrors": []}, schema={"column": pl.String, "nerrors": pl.Int64})
        nrows, ncols = df.shape
    else:
        # Pandas: different handling
        if drop_columns:
            # For constant column detection
            if "numeric" in kind:
                # Numeric constants: min == max. Per-column loop measured faster than
                # a single df.agg(['min','max']) pass on both narrow-tall and wide-short
                # shapes (benchmark 2026-04-14: 10x slower on 1000x200, 0.74x on 100k x 50).
                # NaN-skipping min/max preserves Polars semantics ([1.0, NaN, 1.0] constant).
                constant_cols = [col for col in df.select_dtypes(include="number").columns if df[col].min() == df[col].max()]
            else:
                # Categorical constants: n_unique == 1
                constant_cols = [col for col in df.select_dtypes(exclude="number").columns if df[col].nunique() == 1]
            errors_df = pd.DataFrame({"column": constant_cols, "nerrors": [1] * len(constant_cols)})
        else:
            # For NaN/null/inf detection - use vectorized operations
            numeric_df = df.select_dtypes(include="number")
            if "NaN" in kind:
                nerrors_series = numeric_df.isna().sum()
            elif "null" in kind:
                nerrors_series = numeric_df.isnull().sum()
            elif "infinite" in kind:
                nerrors_series = np.isinf(numeric_df).sum()
            else:
                nerrors_series = pd.Series(dtype=int)

            # Filter to non-zero and convert to DataFrame
            nerrors_series = nerrors_series[nerrors_series > 0].sort_values(ascending=False)
            errors_df = pd.DataFrame({"column": nerrors_series.index.tolist(), "nerrors": nerrors_series.values})
        nrows, ncols = df.shape

    # Log and handle errors
    if len(errors_df) > 0:
        if verbose:
            logger.info(f"Found {len(errors_df)} {kind} out of {ncols} columns:")
            logger.info(f"\n{errors_df}")

        if drop_columns:
            cols_to_drop = errors_df["column"].to_list() if is_polars else errors_df["column"].tolist()
            if cols_to_drop:
                # Only drop columns that actually exist in the dataframe
                if is_polars:
                    existing_cols = filter_existing(df, cols_to_drop)
                    if existing_cols:
                        df = df.drop(existing_cols)
                else:
                    existing_cols = filter_existing(df, cols_to_drop)
                    if existing_cols:
                        df = df.drop(columns=existing_cols)
            if verbose:
                logger.info(f"Dropped {len(cols_to_drop)} {kind}.")
        elif fill_value is not None:
            if is_polars:
                if fill_func_name is None:
                    raise ValueError("fill_func_name is required for Polars DataFrames when fill_value is provided")
                if fill_func_name == "replace":
                    # For infinities: only ±inf are replaced. NaN is preserved here
                    # (cs.numeric().replace does not touch NaN). Callers that also want
                    # NaN filled should run `process_nans` first.
                    df = df.with_columns(cs.numeric().replace([float("inf"), float("-inf")], fill_value))
                else:
                    df = df.with_columns(getattr(cs.numeric(), fill_func_name)(fill_value))
            else:
                if "NaN" in kind or "null" in kind:
                    df = df.fillna(fill_value)
                elif "infinite" in kind:
                    df = df.replace([float("inf"), float("-inf")], fill_value)
            if verbose:
                logger.info(f"{kind.capitalize()}s filled with {fill_value} value.")
                log_ram_usage()

    return df


def process_nans(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Process NaN values in numeric columns by filling them with a specified value.

    Args:
        df: Polars or pandas DataFrame to process
        fill_value: Value to replace NaNs with (default: 0.0)
        verbose: Verbosity level (default: 1)

    Returns:
        DataFrame with NaN values filled
    """
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_nan().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="fill_nan",
        kind="NaN",
        fill_value=fill_value,
        verbose=verbose,
    )


def process_nulls(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Process NULL values in numeric columns by filling them with a specified value.

    Args:
        df: Polars or pandas DataFrame to process
        fill_value: Value to replace NULLs with (default: 0.0)
        verbose: Verbosity level (default: 1)

    Returns:
        DataFrame with NULL values filled
    """
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_null().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="fill_null",
        kind="null",
        fill_value=fill_value,
        verbose=verbose,
    )


def process_infinities(df: Union[pl.DataFrame, pd.DataFrame], fill_value: float = 0.0, verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Process infinite values in numeric columns by replacing them with a specified value.

    Args:
        df: Polars or pandas DataFrame to process
        fill_value: Value to replace infinities with (default: 0.0)
        verbose: Verbosity level (default: 1)

    Returns:
        DataFrame with infinite values replaced
    """
    return _process_special_values(
        df=df,
        expr_func=lambda: (cs.numeric().is_infinite().sum() if isinstance(df, pl.DataFrame) else None),
        fill_func_name="replace",
        kind="infinite",
        fill_value=fill_value,
        verbose=verbose,
    )


def get_numeric_columns(df: Union[pl.DataFrame, pd.DataFrame]) -> list:
    """
    Get list of numeric column names from DataFrame schema without scanning data.

    Args:
        df: Polars or pandas DataFrame

    Returns:
        List of numeric column names

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
    >>> sorted(get_numeric_columns(df))
    ['a', 'c']
    """
    if isinstance(df, pl.DataFrame):
        return [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    else:
        return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: Union[pl.DataFrame, pd.DataFrame], include_string: bool = True) -> list:
    """
    Get list of categorical column names from DataFrame schema without scanning data.

    Args:
        df: Polars or pandas DataFrame
        include_string: Whether to include string columns (default True)

    Returns:
        List of categorical column names
    """
    if isinstance(df, pl.DataFrame):
        # Function-local import: strategies imports from utils, so a top-level
        # import here would form a strategies→utils→strategies cycle at module load.
        from .strategies import get_polars_cat_columns
        if include_string:
            return get_polars_cat_columns(df)
        else:
            return [name for name, dtype in df.schema.items() if dtype == pl.Categorical]
    else:
        # Function-local import (see note above) — breaks strategies↔utils cycle.
        from .strategies import PANDAS_CATEGORICAL_DTYPES
        if include_string:
            return df.select_dtypes(include=list(PANDAS_CATEGORICAL_DTYPES)).columns.tolist()
        else:
            return df.select_dtypes(include=["category"]).columns.tolist()


def remove_constant_columns(df: Union[pl.DataFrame, pd.DataFrame], verbose: int = 1) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Remove constant columns from DataFrame.

    Args:
        df: Input DataFrame
        verbose: Verbosity level

    Returns:
        DataFrame with constant columns removed

    Notes:
        - Numeric columns: where min == max
        - Non-numeric columns: where n_unique == 1
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        # Process numeric columns (min == max)
        df = _process_special_values(
            df=df,
            expr_func=lambda: (cs.numeric().min() == cs.numeric().max()),
            kind="constant numeric columns",
            drop_columns=True,
            verbose=verbose,
        )

        # Process non-numeric columns (n_unique == 1)
        df = _process_special_values(
            df=df,
            expr_func=lambda: (cs.by_dtype(pl.String, pl.Categorical).n_unique() == 1),
            kind="constant non-numeric columns",
            drop_columns=True,
            verbose=verbose,
        )
    else:
        # Pandas: match Polars semantics (min==max for numeric; n_unique==1 for others).
        # Per-column loop measured faster than df.agg(['min','max']) — see audit 2026-04-14.
        numeric_cols = df.select_dtypes(include="number").columns
        constant_num_cols = [col for col in numeric_cols if df[col].min() == df[col].max()]

        # All-NaN numeric columns: min==max yields NaN==NaN -> False, so they are NOT
        # flagged by the loop above. Treat them as constant too (no information).
        all_nan_num = [c for c in numeric_cols if c not in constant_num_cols and df[c].isna().all()]
        constant_num_cols = constant_num_cols + all_nan_num

        non_numeric_cols = df.select_dtypes(exclude="number").columns
        constant_cat_cols = []
        for col in non_numeric_cols:
            try:
                if df[col].nunique(dropna=False) <= 1:
                    constant_cat_cols.append(col)
            except TypeError:
                # Unhashable values (e.g. list/np.ndarray embeddings) — can't be constant-checked.
                continue

        constant_cols = list(constant_num_cols) + list(constant_cat_cols)

        if constant_cols and verbose:
            logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")

        if constant_cols:
            df = df.drop(columns=constant_cols)

    return df


__all__ = [
    "log_ram_usage",
    "log_phase",
    "clean_ram_and_gpu",
    "estimate_df_size_mb",
    "get_process_rss_mb",
    "should_clean_ram",
    "maybe_clean_ram_and_gpu",
    "drop_columns_from_dataframe",
    "get_pandas_view_of_polars_df",
    "save_series_or_df",
    "process_nans",
    "process_nulls",
    "process_infinities",
    "get_numeric_columns",
    "get_categorical_columns",
    "remove_constant_columns",
    "filter_existing",
]
