"""
Utility functions for mlframe training pipeline.

Functions for RAM management, file I/O, dataframe conversions, and data cleaning.
"""
from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
import re
import sys
from textwrap import shorten
from typing import Any, Union, Optional

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)



from ._ram_helpers import _caller_logger, log_ram_usage, maybe_clean_ram_adaptive, clean_ram_and_gpu, estimate_df_size_mb, get_process_rss_mb, should_clean_ram, maybe_clean_ram_and_gpu  # noqa: E402,F401
from pyutilz.system import get_own_memory_usage


def coerce_to_numpy(arr, *, allow_none: bool = False):
    """Coerce a pandas/polars/array-like value to a numpy ndarray without reshaping.

    Single source of truth for the 3-way coercion previously duplicated in
    ``drift_report.py``, ``baseline_diagnostics.py`` and ``composite_estimator.py``.
    Keeps shape intact (the per-module duplicates differed only in whether they
    reshaped to 1-D; reshape lives in ``coerce_to_1d_numpy`` below).

    Parameters
    ----------
    arr
        Anything with ``.to_numpy()`` (polars / pandas), ``.values`` (legacy pandas),
        or numpy/list/scalar.
    allow_none
        When True, ``None`` passes through unchanged. Default raises ``TypeError`` so
        accidental ``None`` inputs are caught at the boundary instead of producing
        opaque downstream errors.
    """
    if arr is None:
        if allow_none:
            return None
        raise TypeError("coerce_to_numpy: input is None; pass allow_none=True to opt in")
    # Fast-path: already an ndarray. Skip the .to_numpy() round-trip (numpy doesn't have a .to_numpy method but
    # pyutilz wrappers occasionally do). Saves a no-op copy on hot loops that pass ndarrays in.
    if isinstance(arr, np.ndarray):
        return arr
    # Fast-path for pandas Series: .values is a zero-copy view of the underlying ndarray
    # (when dtype is numeric / object), strictly cheaper than .to_numpy() which always
    # honours its copy=False default with an extra dtype-resolution step.
    if isinstance(arr, pd.Series):
        return arr.values
    # Fast-path for polars Series: explicit zero_copy_only hint avoids
    # the materialise-via-Arrow step polars takes when called with no kwargs.
    if pl is not None and isinstance(arr, pl.Series):
        try:
            return arr.to_numpy(zero_copy_only=False)
        except TypeError:
            # Older polars without zero_copy_only kwarg.
            return arr.to_numpy()
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    if hasattr(arr, "values"):
        return arr.values
    return np.asarray(arr)


def coerce_to_1d_numpy(arr) -> np.ndarray:
    """Coerce to numpy then flatten to 1-D. Raises ``TypeError`` on ``None``."""
    return np.asarray(coerce_to_numpy(arr)).reshape(-1)


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

    Width 80 reads comfortably on terminals and in notebook cells (was 160 --
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
    df: pd.DataFrame | pl.DataFrame,
    additional_columns_to_drop: list | None = None,
    config_drop_columns: list | None = None,
    verbose: int = 1,
) -> pd.DataFrame | pl.DataFrame:
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

    # Guard isinstance with `pl is not None` -- the module's top-level
    # `import polars as pl` is wrapped in try/except (line 27) so `pl`
    # can be None on installs without polars; isinstance(df, None) raises.
    if pl is not None and isinstance(df, pl.DataFrame):
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
    category palettes (e.g. union vs partial) produce different hashes --
    that's the correct signal for Fix 8's fingerprint, since CatBoost's
    internal dict ordering depends on the category list.

    CACHE-Low-3: explicit coverage for List / Struct / Datetime / Duration so
    minor polars-version repr drift does not change the canonical form.
    Datetime / Duration records both the time unit and timezone (datetime
    only) so naive vs tz-aware columns hash to distinct values.
    """
    s = str(dtype)
    # pl.String is an alias for pl.Utf8 as of polars 1.x; collapse.
    if s in ("Utf8", "String"):
        return "String"
    # pl.Enum(categories=['a','b','c']) -- include categories in canonical form
    # so val-drift with new categories invalidates cached models.
    if s.startswith("Enum("):
        try:
            cats = list(dtype.categories) if hasattr(dtype, "categories") else []
            return "Enum[" + ",".join(sorted(str(c) for c in cats)) + "]"
        except Exception:
            return s
    # pl.List(inner) -- record inner dtype recursively so List[Int64] != List[Float64].
    inner = getattr(dtype, "inner", None)
    if inner is not None and s.startswith("List"):
        return "List[" + _canonical_dtype_str(inner) + "]"
    # pl.Struct(fields=[...]) -- record field name + dtype pairs sorted by name.
    if s.startswith("Struct"):
        fields = getattr(dtype, "fields", None)
        if fields is not None:
            try:
                parts = sorted(
                    f"{getattr(f, 'name', '?')}:{_canonical_dtype_str(getattr(f, 'dtype', '?'))}"
                    for f in fields
                )
                return "Struct{" + ",".join(parts) + "}"
            except Exception:
                return s
    # pl.Datetime(time_unit='us', time_zone='UTC') -- include unit + tz so
    # naive vs UTC-aware columns differ.
    if s.startswith("Datetime"):
        tu = getattr(dtype, "time_unit", None)
        tz = getattr(dtype, "time_zone", None)
        if tu is not None or tz is not None:
            return f"Datetime[{tu or '?'},{tz or 'naive'}]"
        return s
    # pl.Duration(time_unit='us') -- include unit so ms vs us hash differently.
    if s.startswith("Duration"):
        tu = getattr(dtype, "time_unit", None)
        if tu is not None:
            return f"Duration[{tu}]"
        return s
    return s


def compute_model_input_fingerprint(
    df_at_fit,
    cat_features: list | None = None,
    text_features: list | None = None,
    embedding_features: list | None = None,
    *,
    target_name: str | None = None,
    preprocessing_config: Any = None,
    pipeline_config: Any = None,
    model_family: str | None = None,
    random_seed: int | None = None,
    train_idx: Any = None,
    val_idx: Any = None,
) -> tuple[str, list]:
    """Compute a 10-char SHA256 fingerprint of a model's fit-time input
    schema PLUS the surrounding training context.

    Returns:
        (schema_hash, input_schema) where input_schema is the
        order-stable list of ``{"name": str, "dtype": str, "role":
        cat|text|embedding|numeric}`` records used to build the hash.

    The hash now folds in: column row count, target column name, a
    digest of the preprocessing config, a digest of the pipeline config,
    model family, random_seed, and a digest of the train/val split
    indices. Previously the hash was schema-only, so two runs that
    differed in (e.g.) target column or preprocessing would collide on
    the same model filename. The extra fields close that gap.

    Canonical JSON via ``json.dumps(..., sort_keys=True)`` -- required by
    the user's memory rule ``feedback_json_hash_sort_keys`` so the hash
    is deterministic across Python builds with different dict orderings.
    """
    import hashlib
    import json as _json

    if df_at_fit is None:
        # CACHE-Low-1: marker length must equal SHA-256 prefix length (10) so
        # downstream `len(fingerprint) == 10` invariants hold.
        return "__nodf____", []

    cat_set = set(cat_features or [])
    text_set = set(text_features or [])
    emb_set = set(embedding_features or [])

    schema = []
    for col in sorted(df_at_fit.columns):
        try:
            if pl is not None and isinstance(df_at_fit, pl.DataFrame):
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

    # Extra context dimensions. None is serialised stably; pydantic configs
    # are reduced via ``model_dump``; ndarrays / lists are summarised by
    # (length, first/middle/last) to keep the hash O(1).
    def _idx_digest(idx) -> str:
        if idx is None:
            return "none"
        try:
            arr = np.asarray(idx).ravel()
            n = int(arr.size)
            if n == 0:
                return "empty"
            picks = [int(arr[0]), int(arr[n // 2]), int(arr[-1])]
            return f"n{n}:{picks[0]}:{picks[1]}:{picks[2]}"
        except Exception:
            return f"unhashable_{type(idx).__name__}"

    def _config_digest(cfg) -> str:
        if cfg is None:
            return "none"
        try:
            if hasattr(cfg, "model_dump"):
                payload = cfg.model_dump(mode="json")
            elif hasattr(cfg, "dict"):
                payload = cfg.dict()
            elif isinstance(cfg, dict):
                payload = cfg
            else:
                payload = repr(cfg)
            return hashlib.blake2b(
                _json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
                digest_size=8,
            ).hexdigest()
        except Exception:
            return "uncached"

    try:
        n_rows = int(len(df_at_fit))
    except Exception:
        n_rows = -1

    payload = {
        "schema": schema,
        "n_rows": n_rows,
        "target_name": target_name,
        "preprocessing_config": _config_digest(preprocessing_config),
        "pipeline_config": _config_digest(pipeline_config),
        "model_family": model_family,
        "random_seed": random_seed,
        "train_idx": _idx_digest(train_idx),
        "val_idx": _idx_digest(val_idx),
    }

    canonical = _json.dumps(payload, sort_keys=True, default=str)
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
        - Earlier versions cast dict->string here. That was ~37% slower
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
        cross-call reuse -- net speedup was zero. See the 2026-04-19 CHANGELOG
        entry for the full investigation. If production-grade dict sharing is
        ever needed, the right primitive is ``pl.Enum`` on the source column
        (fixed domain preserved across slices), not a post-hoc Arrow-level
        cache.
    """
    if pl is None or not isinstance(df, (pl.DataFrame, pl.Series)):
        raise TypeError(f"Input must be a Polars DataFrame or Series, got {type(df).__name__}")

    import pyarrow as pa

    # Capture timing + RAM around the conversion. Long conversions (multi-GB
    # frames) on prod were silent black boxes -- operators couldn't tell whether
    # the 35-min stall was here or downstream. Log when the conversion crosses
    # log_threshold_seconds so small bridge calls don't spam the log.
    import time as _time
    _t0 = _time.perf_counter()
    _rss_before_gb = get_own_memory_usage()
    _shape_str = f"{df.shape[0]:_}x{df.shape[1]}" if (pl is not None and isinstance(df, pl.DataFrame)) else f"{len(df):_}"

    # Diagnostic: warn on nested Polars types that pyarrow's default
    # ``to_pandas()`` materializes as ``object`` dtype with Python list/
    # dict elements (2026-04-19 probe finding). Downstream CatBoost's
    # embedding_features fastpath expects numeric vectors, not object
    # dtype with list elements, and fails at model.fit() with an opaque
    # "expected numeric" error. Warn here so the operator traces back
    # to the bridge step rather than CatBoost internals. We don't
    # raise or auto-cast -- the bridge is a general-purpose helper and
    # other callers (logging, post-hoc analysis) legitimately want
    # list-typed columns pass-through.
    #
    # Noise-dedupe (2026-04-19 round-11): the bridge is called many
    # times per training run (train / val / test x per-model), and a
    # frame with embedding features would emit the same WARN every call.
    # Fire at most once per unique (column-name, dtype-str) tuple set
    # observed in the process -- repeated identical schemas stay quiet.
    if pl is not None and isinstance(df, pl.DataFrame):
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
                # Trim the displayed list: at most 3 columns, and each
                # column's dtype repr is collapsed if it exceeds 80 chars
                # (a single ``pl.Enum`` over 87 ontology categories serializes
                # to a multi-KB string and floods the log -- one user's run
                # showed "Enum(categories=['3D Modeling & CAD', 'AI & Machine
                # Learning', 'Accounting & Bookkeeping', ...])" running for
                # hundreds of category names per column). The full list is
                # still recoverable from ``df.schema`` if an operator wants
                # to inspect it; this warning is a hint, not a manifest.
                _MAX_DTYPE_REPR = 80
                _MAX_COLS_SHOWN = 3
                def _truncate_dtype(s: str) -> str:
                    if len(s) <= _MAX_DTYPE_REPR:
                        return s
                    return s[:_MAX_DTYPE_REPR - 5] + "...)"
                shown = [(n, _truncate_dtype(t)) for n, t in nested_cols[:_MAX_COLS_SHOWN]]
                if len(nested_cols) > _MAX_COLS_SHOWN:
                    shown_repr = "%s (+%d more)" % (shown, len(nested_cols) - _MAX_COLS_SHOWN)
                else:
                    shown_repr = repr(shown)
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
                    len(nested_cols), shown_repr,
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
    # Python True/False/None elements -- verified empirically 2026-04-23 against
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
    # CRITICAL: to_pandas() defaults are NOT zero-copy -- they CONSOLIDATE memory
    # blocks across columns of the same dtype, forcing a full copy of every
    # numeric buffer into a fresh numpy array. On 9M x 70 numeric columns this
    # was ~35 minutes of pure memcpy + GIL churn in production.
    #   * split_blocks=True  -> no consolidation; each column keeps its Arrow
    #     buffer, so numeric/bool dtypes become np.ndarray views instead of
    #     copies. Categorical still allocates because pandas needs its own
    #     codes representation, but that's the only unavoidable copy.
    #   * use_threads=True   -> parallel column materialization (default but
    #     stated explicitly so the contract is visible).
    #   * self_destruct=False (default) -> SAFE. Tested 2026-04-22: enabling
    #     self_destruct=True caused a native crash inside pytest during the
    #     integration suite -- pyarrow's "EXPERIMENTAL" warning was no joke.
    #     The 26% extra speedup is not worth the segfault risk; opt-in only.
    # Bench (2026-04-22, 7.3M x 118 with 18 dict cols):
    #   default                          30.06s   (1x)
    #   use_threads only                  2.10s   (14x)
    #   +split_blocks                     0.95s   (32x)  <- default
    #   +self_destruct                    0.70s   (43x)  <- opt-in, may crash
    pandas_df = tbl_fixed.to_pandas(
        use_threads=True,
        split_blocks=True,
        self_destruct=self_destruct,
    )

    # Coerce object-materialized nullable Boolean -> pandas Int8 with ``pd.NA``.
    # Verified 2026-04-23 that Int8/``pd.NA`` is accepted by all three tree
    # backends (LightGBM 4.6, XGBoost 3.2, CatBoost 1.2.10); pandas nullable
    # ``boolean`` is rejected by CatBoost ("Cannot convert <NA> to float") so we
    # avoid it. Non-null Boolean already came out as numpy ``bool`` and is left
    # alone. Object-elsewhere (e.g. ``pl.List`` / ``pl.Struct``) is untouched --
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
            "get_pandas_view_of_polars_df %s: %.2fs, RAM %.2f->%.2f GB (delta%+.2f), "
            "self_destruct=%s",
            _shape_str, _elapsed, _rss_before_gb, _rss_after_gb,
            _rss_after_gb - _rss_before_gb, self_destruct,
        )

    return pandas_df


def save_series_or_df(
    obj: pd.Series | pd.DataFrame | pl.Series | pl.DataFrame,
    file: str,
    compression: str = "zstd",
    name: str | None = None,
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

    _series_types = (pd.Series,) if pl is None else (pd.Series, pl.Series)
    if isinstance(obj, _series_types):
        if name:
            obj = obj.to_frame(name=name)
        else:
            obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(file, compression=compression)
    elif pl is not None and isinstance(obj, pl.DataFrame):
        obj.write_parquet(file, compression=compression)


from ._nan_processing import _process_special_values, process_nans, process_nulls, process_infinities, get_numeric_columns, get_categorical_columns, remove_constant_columns  # noqa: E402,F401
