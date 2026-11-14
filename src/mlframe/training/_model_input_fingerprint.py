"""The fingerprint of a model's input schema: dtype families, the canonical dtype string, the per-column schema digest and ``compute_model_input_fingerprint``.

Carved out of ``mlframe.training.utils`` (which re-exports every name here) to keep that module under the line budget.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.utils")


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
        except Exception as e:
            logger.debug("failed to read Enum categories for canonical dtype string, falling back to str(dtype): %s", e)
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
            except Exception as e:
                logger.debug("failed to build canonical Struct field digest, falling back to str(dtype): %s", e)
                return s
    # pl.Datetime(time_unit='us', time_zone='UTC') -- include unit + tz so
    # naive vs UTC-aware columns differ.
    if s.startswith("Datetime"):
        tu = getattr(dtype, "time_unit", None)
        tz = getattr(dtype, "time_zone", None)
        if tu is not None or tz is not None:
            _tu_label = tu if tu else "?"
            _tz_label = tz if tz else "naive"
            return f"Datetime[{_tu_label},{_tz_label}]"
        return s
    # pl.Duration(time_unit='us') -- include unit so ms vs us hash differently.
    if s.startswith("Duration"):
        tu = getattr(dtype, "time_unit", None)
        if tu is not None:
            return f"Duration[{tu}]"
        return s
    return s


def _fingerprint_schema(df_at_fit, cat_features, text_features, embedding_features) -> list:
    """Order-stable ``{"name", "dtype", "role"}`` records for every column of ``df_at_fit``."""
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
        except Exception as e:
            log_throttle(
                logger,
                "utils.compute_model_input_fingerprint",
                logging.WARNING,
                "failed to read dtype for column %r, using placeholder '?' in the fingerprint schema: %s",
                col,
                e,
            )
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
    return schema


def _idx_digest(idx) -> str:
    """Digest an index-like array as ``n<size>:<blake2b of the full content>``."""
    import hashlib

    if idx is None:
        return "none"
    try:
        arr = np.asarray(idx).ravel()
        n = int(arr.size)
        if n == 0:
            return "empty"
        if arr.dtype.kind in "biuf":
            raw = np.ascontiguousarray(arr).tobytes()
        else:
            raw = "\x1f".join(map(str, arr.tolist())).encode("utf-8", "backslashreplace")
        return f"n{n}:{hashlib.blake2b(raw, digest_size=8).hexdigest()}"
    except Exception as e:
        logger.debug("failed to digest index-like object of type %s for the fit fingerprint: %s", type(idx).__name__, e)
        return f"unhashable_{type(idx).__name__}"


def _config_digest(cfg) -> str:
    """Reduce a config object (pydantic model, dict, or arbitrary object) to a stable blake2b digest over its sorted-key JSON payload."""
    import hashlib

    from mlframe.training._canonical_json import canonical_json_bytes

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
    except Exception as e:
        logger.debug("failed to dump a config object for the fit fingerprint, digesting its repr instead: %s", e)
        payload = repr(cfg)
    return hashlib.blake2b(canonical_json_bytes(payload), digest_size=8).hexdigest()


def compute_model_input_fingerprint(
    df_at_fit: pd.DataFrame | pl.DataFrame,
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

    Args:
        df_at_fit: Frame the model is fit on; its columns, dtypes and row count feed the schema hash.
        cat_features: Names of categorical feature columns.
        text_features: Names of text feature columns.
        embedding_features: Names of embedding feature columns.
        target_name: Target column name folded into the hash.
        preprocessing_config: Preprocessing config whose digest is folded into the hash.
        pipeline_config: Pipeline config whose digest is folded into the hash.
        model_family: Model family label folded into the hash.
        random_seed: Random seed folded into the hash.
        train_idx: Train split indices folded into the hash.
        val_idx: Validation split indices folded into the hash.

    Returns:
        (schema_hash, input_schema) where input_schema is the
        order-stable list of ``{"name": str, "dtype": str, "role":
        cat|text|embedding|numeric}`` records used to build the hash.

    The hash folds in the schema, row count and whichever of the optional context arguments the caller passes: target column name,
    preprocessing / pipeline config digests, model family, random_seed and the train/val split indices. Callers that pass only the
    feature lists get a schema-level hash; the context arguments are not filled in implicitly.

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

    schema = _fingerprint_schema(df_at_fit, cat_features, text_features, embedding_features)

    # Extra context dimensions. None is serialised stably; pydantic configs are reduced via ``model_dump``;
    # ndarrays / lists are digested over their full content so distinct splits never collide.
    try:
        n_rows = len(df_at_fit)
    except Exception as e:
        logger.debug("failed to compute len(df_at_fit) for the fit fingerprint, using placeholder -1: %s", e)
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
