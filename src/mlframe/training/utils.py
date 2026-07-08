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
from textwrap import shorten
from typing import Any

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
    # Fast-path for polars Series: explicit allow_copy=True keeps the
    # Arrow-materialise step available (nullable / chunked Arrow series need
    # the copy; disallowing it would raise). polars 0.20.10 renamed the kwarg
    # from zero_copy_only (inverse) to allow_copy (positive); the try/except
    # cascade covers the rename for older builds.
    if pl is not None and isinstance(arr, pl.Series):
        try:
            return arr.to_numpy(allow_copy=True)
        except TypeError:
            try:
                return arr.to_numpy(zero_copy_only=False)
            except TypeError:
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

    _cols_to_drop_list = []
    if additional_columns_to_drop:
        _cols_to_drop_list.extend(additional_columns_to_drop)
    if config_drop_columns:
        _cols_to_drop_list.extend(config_drop_columns)

    # Remove duplicates
    all_cols_to_drop = set(_cols_to_drop_list)

    if not all_cols_to_drop:
        return df

    if verbose:
        logger.info("Dropping %d column(s): %s...", len(all_cols_to_drop), shorten(",".join(all_cols_to_drop), 250))

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


# Polars nested-dtype classes that pyarrow's ``to_pandas()`` materialises as
# pandas ``object`` dtype with Python list/dict elements. Resolved once at
# import (defensively -- older polars may lack ``Array``) so the bridge's
# nested-column scan can use ``isinstance`` instead of building a per-column
# ``str(dt)`` (which is multi-KB for wide Enum/Categorical dtypes).
_NESTED_PL_DTYPES: tuple = tuple(getattr(pl, _n) for _n in ("List", "Array", "Struct", "Object") if hasattr(pl, _n)) if pl is not None else ()


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
                parts = sorted(f"{getattr(f, 'name', '?')}:{_canonical_dtype_str(getattr(f, 'dtype', '?'))}" for f in fields)
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

    Canonical JSON via ``orjson.dumps(..., option=OPT_SORT_KEYS)`` -- required
    by the user's memory rule ``feedback_json_hash_sort_keys`` so the hash is
    deterministic across Python builds with different dict orderings.
    """
    import hashlib
    import orjson

    if df_at_fit is None:
        # CACHE-Low-1: marker length must equal SHA-256 prefix length (10) so
        # downstream `len(fingerprint) == 10` invariants hold.
        return "__nodf____", []

    cat_set = set(cat_features or [])
    text_set = set(text_features or [])
    emb_set = set(embedding_features or [])

    schema = []
    # Sort by ``str(col)`` so a frame carrying a mix of int-positional and string column labels
    # (engineered string-named columns alongside raw integer-indexed ones) does not raise
    # ``TypeError: '<' not supported between instances of 'int' and 'str'`` on the bare sort.
    for col in sorted(df_at_fit.columns, key=str):
        try:
            if pl is not None and isinstance(df_at_fit, pl.DataFrame):
                dt: Any = df_at_fit.schema[col]
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
                orjson.dumps(payload, default=str, option=orjson.OPT_SORT_KEYS),
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

    canonical = orjson.dumps(payload, default=str, option=orjson.OPT_SORT_KEYS)
    digest = hashlib.sha256(canonical).hexdigest()[:10]
    return digest, schema


# iter628 (perf): single-entry "last result" memo for get_pandas_view_-
# of_polars_df. Key = (id(pl_df), shape); value = the previously-
# returned pandas DataFrame. Module-level so it survives across
# call-site boundaries; single-entry so it's bounded memory (the
# cached pandas view holds a ref on the polars buffer, so we don't
# want to retain a long history). Matches the iter625 / iter627
# "double-call with same id in tight succession" doctrine.
_PD_VIEW_LAST_CACHE: dict = {"id_key": None, "result": None}


def clear_pandas_view_cache() -> None:
    """Drop the single-entry ``get_pandas_view_of_polars_df`` memo. The cached pandas view is Arrow-backed and shares
    the polars frame's buffers zero-copy, so while it lives it PINS those buffers. Call this when releasing ctx polars
    frames to reclaim RAM -- otherwise the last-converted frame's gigabytes stay resident behind the memo and the
    release shows a ~0 MB delta (observed prod 2026: 8 GB expected, 0 reclaimed)."""
    _PD_VIEW_LAST_CACHE["id_key"] = None
    _PD_VIEW_LAST_CACHE["result"] = None


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

    # iter628 (perf): single-entry "last result" memo. c0008 @100k
    # profile showed 8 calls / 8.08s cumtime to this bridge across
    # multiple callers (_normalise_X dominates at 7.44s; the other 7
    # share 0.64s). Sequential calls from neighbouring code blocks
    # often pass the SAME polars frame (train/val/test slices re-used
    # across model strategies). The memo keyed by (id(df), shape,
    # self_destruct) returns the cached pandas view on immediate
    # repeats without re-running the ~1s pyarrow conversion.
    #
    # Safety: pl.DataFrame is mutable so id() recycling after GC is
    # the standard risk; shape inclusion gives an extra discriminator.
    # self_destruct=True paths bypass the memo (consuming input is
    # destructive; never returning a cached prior view for those).
    # The returned pandas view shares memory with the polars buffer,
    # so two callers receiving the same cached view will see each
    # other's in-place mutations. mlframe consumers don't mutate the
    # bridge output (verified across the 5 call sites:
    # _normalise_X / _prepare_strategy_inputs / _cb_polars_to_pandas
    # / _coerce_to_pandas / _to_pandas_for_baseline) -- all read-only.
    # If a future caller does mutate, the memo here is the first
    # place to look for the bug.
    if not self_destruct:
        sh = getattr(df, "shape", None)
        # Co-validate column names alongside id()+shape: id() recycles after GC, so a different frame with
        # the SAME shape could otherwise false-hit and return a stale pandas view; the column tuple is a cheap
        # extra discriminator that catches the common same-shape-different-columns recycle.
        _cols = tuple(df.columns) if hasattr(df, "columns") else None
        _id_key = (id(df), sh if sh is not None else (None,), _cols)
        _cached = _PD_VIEW_LAST_CACHE.get("id_key")
        if _cached == _id_key:
            _result = _PD_VIEW_LAST_CACHE.get("result")
            if _result is not None:
                return _result

    # iter354 (2026-05-27) size-aware dispatcher: the pyarrow-Table +
    # per-column-cast path below is 9.5x faster than ``df.to_pandas()`` for
    # numeric-heavy frames (200k x 30 float64: 13.3ms -> 1.4ms) but 2.4x
    # SLOWER for SMALL frames dominated by polars Categorical / Enum
    # columns (200k x 15 cat + 10 num: 20.4ms -> 49.7ms). The slow path is
    # the per-column ``pa.compute.cast`` chain that narrows uint32 dict
    # indices to int32 for pyarrow's to_pandas (still refuses uint32 dict
    # indices on pyarrow 21).
    #
    # CRITICAL size constraint: on LARGE frames (multi-GB / 100 GB
    # production loads) the helper's ``split_blocks=True`` path is the
    # whole reason it exists -- it gives zero-copy numpy views over
    # numeric Arrow buffers instead of consolidating them into fresh
    # numpy blocks. ``df.to_pandas()`` consolidates and copies every
    # numeric buffer, blowing memory + wall time on big workloads
    # (the original 9M x 70 numeric frame went 30s vs 0.95s, ~32x).
    #
    # Dispatch policy:
    #   * Force HELPER for any frame above ~50 MB (size proxy: rows *
    #     cols * 8). On big frames the per-column dict cast amortises and
    #     the zero-copy numeric path dominates. This is also exactly the
    #     regime where ``df.to_pandas()`` would OOM the operator.
    #   * Force RAW ``df.to_pandas()`` only for SMALL Categorical-heavy
    #     frames where the cast overhead bites and there's no numeric
    #     buffer to keep zero-copy.
    #   * Override via env var for forensic A/B benchmarks.
    if isinstance(df, pl.DataFrame) and os.environ.get("MLFRAME_FORCE_HELPER_PD_VIEW") != "1":
        _n_rows = df.shape[0]
        _n_cols = df.shape[1]
        _size_bytes_proxy = _n_rows * _n_cols * 8
        _LARGE_FRAME_BYTES = 50 * 1024 * 1024  # 50 MB
        _dict_cols = sum(1 for dt in df.schema.values() if dt == pl.Categorical or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum)))
        # Small + many-dict frames take the raw path; everything else
        # (including all production-scale frames) takes the helper for
        # zero-copy numeric safety.
        if _size_bytes_proxy < _LARGE_FRAME_BYTES and _dict_cols >= 5:
            _raw = df.to_pandas()
            # Apply the SAME nullable-Boolean coercion the helper path does below: a Boolean column with any null
            # materialises as pandas ``object`` (Python True/False/None), which the tree backends reject. The raw
            # fast-path must not silently emit object dtype just because the frame happened to be dict-heavy + small.
            for _bname, _bdt in df.schema.items():
                if _bdt == pl.Boolean and _raw[_bname].dtype == object:
                    _raw[_bname] = _raw[_bname].astype("boolean").astype("Int8")
            return _raw

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
        # Detect nested dtypes by isinstance, NOT str(dt): stringifying a
        # pl.Enum / pl.Categorical column whose universe is large builds a
        # multi-KB repr (a single Enum over 200 categories serialises to a
        # ~16 KB string -- see the truncation note in the warning below), and
        # this bridge runs many times per training run (train/val/test x
        # per-model). The class check touches the dtype object directly with
        # zero string allocation; only the rare columns that ARE nested get
        # stringified, for the warning message. (Micro-bench: the scan over a
        # frame with a wide Enum column dropped 3.4x.)
        nested_cols = []
        for name, dt in df.schema.items():
            if isinstance(dt, _NESTED_PL_DTYPES):
                nested_cols.append((name, str(dt)))
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

    # Remap pl.Categorical -> pl.Enum keyed on the column's own unique
    # values BEFORE the Arrow bridge. polars 1.x's global string-cache
    # leaks ALL categories ever seen by the process into every pl.Categorical
    # column's dictionary, so a column with values {'a', 'b'} in production
    # would deserialise with codes 4, 5, 6 (instead of 0, 1, 2) after a
    # sibling test added 'C', 'B', 'A', 'cat_0' to the cache. pl.Enum is
    # per-Series and ignores the global cache, so its Arrow round-trip
    # produces a clean dictionary with codes starting at 0 and matching the
    # column's actual category universe. The remap is a no-op for non-
    # Categorical columns and adds O(n_unique) per Categorical column
    # (single .unique() pass) on first access. Verified against the
    # ``tests/training/test_utils.py::TestGetPandasViewOfPolarsDF`` cluster
    # which fails under xdist when sibling tests pollute the cache.
    if pl is not None and isinstance(df, pl.DataFrame):
        _cat_remaps = []
        for _name, _dt in df.schema.items():
            if _dt == pl.Categorical:
                _cat_remaps.append(_name)
        if _cat_remaps:
            # Each Enum domain is the column's OWN unique values; drop_nulls so
            # the domain doesn't carry a sentinel (nulls round-trip through Arrow
            # as null codes regardless).
            #
            # With >= 3 Categorical columns, compute every column's uniques in a
            # SINGLE polars collect rather than one collect per column. The
            # per-column ``drop_nulls().unique().to_list()`` loop issued ~2
            # PyLazyFrame.collect calls per column, and on categorical-heavy
            # frames collect dominated this bridge's self-time (sub-profile,
            # 1M rows x 8 cat cols: collect 0.90s = 74%). A single
            # ``select(... .implode())`` collect runs the unique passes in
            # parallel: 1M-row bench 56.0->46.0ms (1.22x) at ncat=3,
            # 182->149ms (1.22x) at ncat=16, bit-identical category sets +
            # string values. The per-column path is kept for 1-2 columns where
            # the single-collect setup is a wash (ncat=1 0.93x, ncat=2 0.99x).
            if len(_cat_remaps) >= 3:
                _uniq_row = df.lazy().select([pl.col(_name).drop_nulls().unique().implode().alias(_name) for _name in _cat_remaps]).collect()
                _exprs = [pl.col(_name).cast(pl.Enum(_uniq_row.get_column(_name)[0].to_list())) for _name in _cat_remaps]
            else:
                _exprs = []
                for _name in _cat_remaps:
                    _ser = df.get_column(_name)
                    _uniques = _ser.drop_nulls().unique().to_list()
                    _exprs.append(pl.col(_name).cast(pl.Enum(_uniques)))
            df = df.with_columns(_exprs)
    tbl = df.to_arrow()

    # Note: short-circuit on "no dictionary columns" was benchmarked 2026-04-14 and
    # delivered only 1.16x on pure-numeric workloads (below 1.2x threshold), so the
    # per-column scan is retained for code uniformity.
    fixed_cols = []
    for col in tbl.columns:
        if pa.types.is_dictionary(col.type) and col.type.index_type != pa.int32():
            # Whole-ChunkedArray cast: ~5.5x faster than the prior per-chunk
            # ``pa.compute.cast(chunk.indices, ...)`` + ``DictionaryArray.from_arrays``
            # rebuild loop on 1M-row x 3-col frames (microbench 2026-05-20:
            # 8.20ms -> 1.48ms). PyArrow's C++ cast dispatches index narrowing
            # once per array instead of once per chunk + Python-level chunk
            # rewrap, and preserves the dictionary buffer by reference. The
            # ``ordered`` flag is threaded explicitly so ordered enums don't
            # silently lose their ordering metadata.
            target_type = pa.dictionary(pa.int32(), col.type.value_type, ordered=col.type.ordered)
            col = pa.compute.cast(col, target_type)
        fixed_cols.append(col)

    tbl_fixed = pa.table(fixed_cols, names=tbl.column_names)

    # Capture which Arrow columns were bool BEFORE to_pandas(). Nullable Boolean
    # columns (those with any null) materialize as pandas ``object`` dtype with
    # Python True/False/None elements -- verified empirically 2026-04-23 against
    # pyarrow 21 on Polars 1.37. This breaks LightGBM's sklearn wrapper, which
    # refuses ``object`` dtypes (``ValueError: pandas dtypes must be int, float
    # or bool``). Non-null Boolean stays as numpy ``bool``. We post-process only
    # the object-materialized ones below.
    _arrow_bool_cols = [name for name, col in zip(tbl_fixed.column_names, tbl_fixed.columns) if pa.types.is_boolean(col.type)]

    # Capture which Arrow columns are date32/date64/timestamp BEFORE to_pandas().
    # On newer pyarrow + pandas combos (verified empirically with pandas 2.x + a
    # str-mapping default), ``to_pandas()`` may materialize ``pa.date32()`` as
    # pandas ``str`` (``StringDtype(na_value=nan)``) instead of ``datetime64[ns]``.
    # Downstream consumers (the dummy-baselines temporal monotonicity helper,
    # ``compute_ml_perf_by_time``) silently take the string path and either
    # raise or fall through with no temporal structure. Coerce back to
    # ``datetime64[ns]`` after the bridge so the documented contract holds.
    _arrow_temporal_cols = [
        name for name, col in zip(tbl_fixed.column_names, tbl_fixed.columns) if (pa.types.is_date(col.type) or pa.types.is_timestamp(col.type))
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

    # Restore datetime64[ns] for any Arrow date / timestamp column the pandas
    # bridge demoted to string / object on this pyarrow version. ``errors="coerce"``
    # keeps malformed cells as NaT rather than blowing the bridge.
    _coerced_temporal_cols = []
    for name in _arrow_temporal_cols:
        if not pd.api.types.is_datetime64_any_dtype(pandas_df[name]):
            _pre_na = int(pandas_df[name].isna().sum())
            pandas_df[name] = pd.to_datetime(pandas_df[name], errors="coerce")
            _new_nat = int(pandas_df[name].isna().sum()) - _pre_na
            if _new_nat > 0:
                logger.warning(
                    "get_pandas_view_of_polars_df: restoring temporal column %r coerced %d "
                    "additional cell(s) to NaT (unparseable timestamps silently dropped).",
                    name, _new_nat,
                )
            _coerced_temporal_cols.append(name)
    if _coerced_temporal_cols and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "get_pandas_view_of_polars_df: restored %d Arrow date/timestamp " "column(s) to datetime64[ns] after pandas bridge demoted them: %s",
            len(_coerced_temporal_cols),
            _coerced_temporal_cols,
        )
    if _coerced_bool_cols and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "get_pandas_view_of_polars_df: coerced %d nullable-bool object " "column(s) to Int8: %s",
            len(_coerced_bool_cols),
            _coerced_bool_cols,
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

    # iter628 (perf): populate the single-entry memo so the NEXT
    # identical-id call returns the cached view. Skip when
    # self_destruct=True (input is being consumed; caching its
    # post-consume id would be unsafe).
    if not self_destruct:
        try:
            sh = getattr(df, "shape", None)
            _cols = tuple(df.columns) if hasattr(df, "columns") else None
            # Publish result BEFORE key so a torn read on this unlocked single-slot memo can only see an OLD key (miss -> recompute), never a NEW id_key
            # paired with a stale pandas view from a prior different df. Key co-validates id()+shape+columns (see the read site).
            _PD_VIEW_LAST_CACHE["result"] = pandas_df
            _PD_VIEW_LAST_CACHE["id_key"] = (id(df), sh if sh is not None else (None,), _cols)
        except Exception:  # nosec B110 - non-trivial body
            # Memo population is best-effort; never fail the conversion
            # because of a cache-write hiccup.
            pass

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
        obj.write_parquet(file, compression=compression)  # type: ignore[arg-type]  # compression is a validated free-form str at the public API boundary; polars narrows to a Literal set at runtime


from ._nan_processing import _process_special_values, process_nans, process_nulls, process_infinities, get_numeric_columns, get_categorical_columns, remove_constant_columns  # noqa: E402,F401
