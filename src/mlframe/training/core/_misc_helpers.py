"""Small utility functions: logging, metrics, DataFrame ops, validation, Polars helpers, dataset reuse detection, tier-DF building."""

from __future__ import annotations

import logging
import sys
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from ..utils import filter_existing

logger = logging.getLogger(__name__)


def _ensure_logging_visible(level: int = logging.INFO) -> None:
    """Install or upgrade a timestamped root handler so mlframe progress logs are visible in Jupyter and plain scripts.

    Replaces non-timestamped formatters in place; leaves handlers that already include ``%(asctime)s`` untouched.
    """
    root = logging.getLogger()
    desired_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    desired_datefmt = "%H:%M:%S"

    # Fast-path: if a previous call already installed an asctime-bearing handler AND the
    # root level is already at or below the requested threshold, there is nothing to do.
    # Mutating handlers on every suite invocation when nothing needs to change makes
    # back-to-back ``train_mlframe_models_suite`` calls re-walk the handler list and
    # re-assign formatters that already satisfy the contract.
    if root.handlers and (root.level != logging.NOTSET and root.level <= level):
        for h in root.handlers:
            existing = getattr(h.formatter, "_fmt", None) if h.formatter else None
            if existing and "%(asctime)" in existing:
                return

    timestamped = logging.Formatter(desired_fmt, datefmt=desired_datefmt)

    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(timestamped)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            existing = getattr(h.formatter, "_fmt", None) if h.formatter else None
            if not existing or "%(asctime)" not in existing:
                h.setFormatter(timestamped)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)


def _entry_metric(entry, split: str, name: str) -> float:
    """Pull a per-split per-name metric value, tolerating legacy nested/flat/split-less shapes; returns NaN on any miss."""
    metrics = getattr(entry, "metrics", None)
    if not isinstance(metrics, dict):
        return float("nan")
    inner = metrics.get(split)
    if isinstance(inner, dict):
        v = inner.get(name)
        if isinstance(v, (int, float)):
            return float(v)
    v = metrics.get(name)
    if isinstance(v, (int, float)):
        return float(v)
    v = metrics.get(f"{split}_{name}")
    if isinstance(v, (int, float)):
        return float(v)
    return float("nan")


def _augment_with_dropped_high_card_cols(
    dropped_data,
    train_df,
    val_df,
    test_df,
    *,
    train_od_idx=None,
    val_od_idx=None,
):
    """Re-attach pre-drop high-card cat columns to ``train/val/test_df``, slicing captured ndarrays by OD-filter indices to row-align.

    Test is never OD-filtered. Returns ``(train_df, val_df, test_df, added_col_names)``.
    """
    added = []
    if not dropped_data:
        return train_df, val_df, test_df, added

    train_extras, val_extras, test_extras = {}, {}, {}
    n_train = len(train_df) if train_df is not None else 0
    n_val = len(val_df) if val_df is not None else 0
    n_test = len(test_df) if test_df is not None else 0

    for col, data in dropped_data.items():
        if "train" in data and train_df is not None:
            arr = data["train"]
            if train_od_idx is not None and len(arr) != n_train:
                arr_aligned = arr[train_od_idx] if len(arr) == len(train_od_idx) else None
            elif len(arr) == n_train:
                arr_aligned = arr
            else:
                arr_aligned = None
            if arr_aligned is not None and len(arr_aligned) == n_train:
                train_extras[col] = arr_aligned
        if "val" in data and val_df is not None:
            arr = data["val"]
            if val_od_idx is not None and len(arr) != n_val:
                arr_aligned = arr[val_od_idx] if len(arr) == len(val_od_idx) else None
            elif len(arr) == n_val:
                arr_aligned = arr
            else:
                arr_aligned = None
            if arr_aligned is not None and len(arr_aligned) == n_val:
                val_extras[col] = arr_aligned
        if "test" in data and test_df is not None:
            arr = data["test"]
            if len(arr) == n_test:
                test_extras[col] = arr
        if col in train_extras:
            added.append(col)

    if not added:
        return train_df, val_df, test_df, added

    def _attach(frame, extras):
        if frame is None or not extras:
            return frame
        if isinstance(frame, pl.DataFrame):
            return frame.with_columns([pl.Series(c, v) for c, v in extras.items()])
        return frame.assign(**extras)

    return (
        _attach(train_df, train_extras),
        _attach(val_df, val_extras),
        _attach(test_df, test_extras),
        added,
    )


def _build_full_column_from_splits(
    col_name,
    train_df,
    val_df,
    test_df,
    train_idx,
    val_idx,
    test_idx,
    n_total,
):
    """Reassemble a single column at the FULL n_total row index space from per-split frames.

    Returns a float64 ndarray of length ``n_total``; rows not covered by any split keep NaN.
    """
    import numpy as _np
    out = _np.full(n_total, _np.nan, dtype=_np.float64)
    for _split_df, _split_idx in (
        (train_df, train_idx), (val_df, val_idx), (test_df, test_idx),
    ):
        if _split_df is None or _split_idx is None:
            continue
        if col_name not in _split_df.columns:
            continue
        try:
            col_vals = _split_df[col_name].to_numpy() if hasattr(_split_df[col_name], "to_numpy") \
                else _np.asarray(_split_df[col_name])
        except Exception:
            continue
        col_vals = _np.asarray(col_vals).reshape(-1).astype(_np.float64, copy=False)
        idx_arr = _np.asarray(_split_idx).reshape(-1)
        if len(col_vals) != len(idx_arr):
            # Frame and index disagree (e.g. OD-filtered train_df paired with raw train_idx); skip rather than mis-align silently.
            continue
        out[idx_arr] = col_vals
    return out


def _drop_cols_df(df, cols):
    """Drop ``cols`` from ``df`` (pandas or Polars), ignoring missing names."""
    import pandas as _pd
    if not cols:
        return df
    existing = filter_existing(df, cols)
    if not existing:
        return df
    if isinstance(df, _pd.DataFrame):
        return df.drop(columns=existing, errors="ignore")
    return df.drop(existing)


def _validate_trusted_path(path: str, trusted_root):
    """Raise ValueError if ``path`` is not inside ``trusted_root``; gates ``joblib.load`` to limit arbitrary-code-execution surface."""
    import os as _os
    if trusted_root is None:
        raise ValueError(
            "trusted_root is required for joblib.load() of metadata files. Pass an "
            "absolute directory under which the metadata artifact is stored."
        )
    abs_root = _os.path.abspath(trusted_root)
    abs_path = _os.path.abspath(path)
    try:
        common = _os.path.commonpath([abs_root, abs_path])
    except ValueError:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}") from None
    if common != abs_root:
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")


def _df_shape_str(df) -> str:
    """Format DataFrame shape as 'rowsxcols' with thousands separators."""
    if df is None:
        return "None"
    nrows = df.shape[0] if hasattr(df, "shape") else len(df)
    ncols = df.shape[1] if hasattr(df, "shape") else 0
    return f"{nrows:_}x{ncols}"


def _elapsed_str(start: float) -> str:
    """Format elapsed time since start as human-readable string."""
    elapsed = timer() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed / 60:.1f}min"


def _detect_dataset_reuse_capabilities() -> dict[str, bool]:
    """Feature-detect which GBDT sklearn wrappers can accept a pre-built dataset as ``X`` for label/weight reuse across fits."""
    caps: dict[str, bool] = {}

    try:
        import catboost as _cb
        _pool_cls = getattr(_cb, "Pool", None)
        caps["cb_pool_set_label"] = callable(getattr(_pool_cls, "set_label", None))
        caps["cb_pool_set_weight"] = callable(getattr(_pool_cls, "set_weight", None))
        caps["cb_pool_label_swap"] = (
            caps["cb_pool_set_label"] and caps["cb_pool_set_weight"]
        )
    except ImportError:
        caps["cb_pool_set_label"] = False
        caps["cb_pool_set_weight"] = False
        caps["cb_pool_label_swap"] = False

    try:
        import xgboost as _xgb
        _dm = getattr(_xgb, "DMatrix", None)
        caps["xgb_dmatrix_set_label"] = callable(getattr(_dm, "set_label", None))
        caps["xgb_dmatrix_set_weight"] = callable(getattr(_dm, "set_weight", None))
        # XGBClassifier.fit(X=DMatrix) does NOT short-circuit yet (xgboost 3.2.0 _create_dmatrix rebuilds unconditionally).
        caps["xgb_sklearn_accepts_dmatrix"] = False
    except ImportError:
        caps["xgb_dmatrix_set_label"] = False
        caps["xgb_dmatrix_set_weight"] = False
        caps["xgb_sklearn_accepts_dmatrix"] = False

    try:
        import lightgbm as _lgb
        _ds = getattr(_lgb, "Dataset", None)
        caps["lgb_dataset_set_label"] = callable(getattr(_ds, "set_label", None))
        caps["lgb_dataset_set_weight"] = callable(getattr(_ds, "set_weight", None))
        # LGBMClassifier.fit(X=Dataset) does NOT short-circuit yet (lightgbm 4.6.0).
        caps["lgb_sklearn_accepts_dataset"] = False
    except ImportError:
        caps["lgb_dataset_set_label"] = False
        caps["lgb_dataset_set_weight"] = False
        caps["lgb_sklearn_accepts_dataset"] = False

    return caps


def _validate_input_columns_against_metadata(
    df,
    metadata: dict[str, Any],
    verbose: bool = False,
):
    """Validate inference-time DataFrame columns against model metadata.

    Missing cat/text/embedding features raise ValueError (cannot be safely dropped); other missing columns WARN + proceed;
    extra columns are dropped (logged when verbose). Returns the possibly-filtered df.
    """
    columns = metadata.get("columns", [])
    if not columns:
        return df

    missing_cols = set(columns) - set(df.columns)
    extra_cols = set(df.columns) - set(columns)

    if missing_cols:
        meta_cat = set(metadata.get("cat_features") or [])
        meta_text = set(metadata.get("text_features") or [])
        meta_emb = set(metadata.get("embedding_features") or [])
        critical_missing = missing_cols & (meta_cat | meta_text | meta_emb)
        if critical_missing:
            raise ValueError(
                f"Input DataFrame is missing {len(critical_missing)} "
                f"load-bearing feature column(s) that the model was "
                f"trained on: {sorted(critical_missing)}. These are "
                f"declared in metadata as cat/text/embedding features; "
                f"the pipeline + model cannot run correctly without "
                f"them. Either restore the upstream extraction that "
                f"produced these columns, or retrain the model on the "
                f"current feature set."
            )
        logger.warning(
            "Missing columns in input: %s. The pipeline will attempt "
            "to proceed -- downstream errors about shape mismatches "
            "usually trace back here.",
            sorted(missing_cols),
        )

    if extra_cols:
        if verbose:
            logger.info("Dropping extra columns: %s", sorted(extra_cols))
        df = df[filter_existing(df, columns)]

    # Per-model input-schema diff: HARD-FAIL on changes that silently corrupt predictions
    # (removed cat/text/embedding cols, role changes, dtype family changes for non-numeric roles);
    # SOFT-WARN on benign differences the pipeline casts transparently (float32<->float64, etc.).
    # Silent pass on old metadata files predating model_schemas.
    model_schemas = metadata.get("model_schemas")
    if model_schemas:
        from mlframe.training.utils import compute_model_input_fingerprint, _dtype_family
        live_hash, live_schema = compute_model_input_fingerprint(
            df,
            cat_features=metadata.get("cat_features") or [],
            text_features=metadata.get("text_features") or [],
            embedding_features=metadata.get("embedding_features") or [],
        )
        live_schema_idx = {entry["name"]: entry for entry in live_schema}
        for model_file_name, rec in model_schemas.items():
            expected_hash = rec.get("schema_hash")
            expected_schema = rec.get("input_schema") or []
            if expected_hash is None or not expected_schema:
                continue
            if expected_hash == live_hash:
                continue
            expected_idx = {entry["name"]: entry for entry in expected_schema}
            # Trained snapshot is POST-pipeline; live snapshot is PRE-pipeline. For cat/text/embedding columns
            # role/dtype is user-declared and stable, so family changes there are critical (silent label-encoding/vocab drift).
            # For numeric-role columns the pipeline casts internally, so family changes are expected -> soft-warn.
            critical_removed: list = []
            family_changes: list = []
            role_changes: list = []
            soft_width_changes: list = []
            soft_family_changes: list = []
            for col, e in expected_idx.items():
                if col not in live_schema_idx:
                    if e["role"] in ("cat", "text", "embedding"):
                        critical_removed.append(col)
                    continue
                live = live_schema_idx[col]
                role_critical = e["role"] in ("cat", "text", "embedding") or live["role"] in ("cat", "text", "embedding")
                if live["role"] != e["role"]:
                    if role_critical:
                        role_changes.append(f"    {col}: trained role={e['role']} serving role={live['role']}")
                if live["dtype"] != e["dtype"]:
                    ef = _dtype_family(e["dtype"])
                    lf = _dtype_family(live["dtype"])
                    if ef != lf:
                        if role_critical:
                            family_changes.append(
                                f"    {col}: trained={e['dtype']!r} ({ef}) serving={live['dtype']!r} ({lf})"
                            )
                        else:
                            soft_family_changes.append(
                                f"    {col}: trained={e['dtype']!r} ({ef}) serving={live['dtype']!r} ({lf}) (numeric role)"
                            )
                    else:
                        soft_width_changes.append(
                            f"    {col}: trained={e['dtype']!r} serving={live['dtype']!r} (same family={lf})"
                        )
            hard_fail = bool(critical_removed or family_changes or role_changes)
            if hard_fail:
                diff_lines = []
                if critical_removed:
                    diff_lines.append(
                        f"  - critical missing (cat/text/embedding): {sorted(critical_removed)}"
                    )
                if family_changes:
                    diff_lines.append("  dtype FAMILY changes (trained -> serving):")
                    diff_lines.extend(family_changes)
                if role_changes:
                    diff_lines.append("  role changes (cat/text/embedding/numeric):")
                    diff_lines.extend(role_changes)
                if soft_width_changes:
                    diff_lines.append("  (soft) dtype width-only changes:")
                    diff_lines.extend(soft_width_changes)
                raise ValueError(
                    "Model input-schema mismatch at load time for "
                    f"{model_file_name!r} "
                    f"(trained hash={expected_hash}, serving hash={live_hash}):\n"
                    + "\n".join(diff_lines) + "\n"
                    "Either restore the upstream feature pipeline that produced "
                    "the trained-time layout, or retrain the model against the "
                    "current serving frame."
                )
            if soft_width_changes or soft_family_changes:
                lines = []
                if soft_width_changes:
                    lines.extend(s.strip() for s in soft_width_changes)
                if soft_family_changes:
                    lines.extend(s.strip() for s in soft_family_changes)
                logger.warning(
                    "Input-schema drift for %s (pipeline-internal casts on "
                    "numeric-role columns and/or width-only changes). "
                    "Accepting; trained pipeline is responsible for "
                    "casting the serving df: %s",
                    model_file_name,
                    "; ".join(lines),
                )

    return df


def _filter_polars_cat_features_by_dtype(
    df: pl.DataFrame,
    cat_features: list[str],
) -> list[str]:
    """Defensive filter for CB Polars fastpath ``cat_features``: keep only Categorical/Enum dtypes.

    CB 1.2.x's Cython fused cpdef dispatcher only matches pl.Categorical (and on some builds pl.Enum); other dtypes
    raise opaque ``TypeError: No matching signature found``. Drops mismatched columns with WARNING; missing columns silently.
    """
    valid: list = []
    dropped: list = []
    for c in cat_features or []:
        if c not in df.columns:
            continue
        dt = df.schema[c]
        is_cat = (dt == pl.Categorical) or (
            hasattr(pl, "Enum") and isinstance(dt, pl.Enum)
        )
        if is_cat:
            valid.append(c)
        else:
            dropped.append((c, str(dt)))
    if dropped:
        logger.warning(
            "Dropping %d column(s) from CB cat_features because their "
            "Polars dtype is not Categorical/Enum: %s. CatBoost's fastpath "
            "dispatcher has no overload for those types and would raise "
            "'No matching signature found'. Most likely cause: the column "
            "was promoted from cat_features to text_features and cast to "
            "pl.String, but the caller is still passing the pre-promotion "
            "list. Fix the caller to use the post-promotion cat_features.",
            len(dropped), dropped,
        )
    return valid


def _auto_detect_feature_types(
    df,
    feature_types_config,
    cat_features: list,
    verbose: bool = False,
) -> tuple:
    """Auto-detect text/embedding features and promote high-cardinality string/categorical columns to text_features.

    Promotion criteria: not user-assigned, dtype is pl.String/pl.Utf8/pl.Categorical (pl.Enum stays nominal),
    n_unique > threshold. Does NOT mutate ``cat_features`` (caller filters via set-difference).

    Returns: ``(text_features, embedding_features, auto_detected_high_card_to_drop)``.
    """
    import polars as pl

    _ftc = feature_types_config
    text_features = list(_ftc.text_features or []) if _ftc is not None else []
    embedding_features = list(_ftc.embedding_features or []) if _ftc is not None else []
    # ``use_text_features=True``: auto-detected cols -> text_features. ``False``: -> auto_detected_high_card_to_drop
    # so caller drops them entirely (prevents XGB QuantileDMatrix OOM and CB artefact bloat on 2M-level cats).
    # User-supplied explicit text_features/embedding_features are honored regardless.
    auto_detected_high_card_to_drop: list = []

    if feature_types_config is None or not feature_types_config.auto_detect_feature_types:
        return text_features, embedding_features, auto_detected_high_card_to_drop

    if cat_features is None:
        cat_features = []

    abs_threshold = feature_types_config.cat_text_cardinality_threshold
    # Minimum non-null FRACTION to promote; below it CB's TF-IDF estimator yields an empty dictionary and raises
    # "Dictionary size is 0" (text_feature_estimators.cpp). Fraction (not count) scales with dataset size.
    min_non_null_frac = getattr(
        feature_types_config, "min_non_null_fraction_for_text_promotion", 0.01
    )
    total_rows = df.height if hasattr(df, "height") else len(df)
    min_non_null_abs = max(1, int(round(total_rows * min_non_null_frac)))
    # Size-aware effective promotion threshold: a flat 300-uniq floor is wrong at both ends of the data-size axis
    # (on 100-row data every string column stays "cat"; on 10M-row data 300 is still a sliver). pct=0 keeps legacy
    # behaviour (effective == absolute). The 50-uniq floor prevents pathologically tiny effective thresholds.
    pct_threshold = getattr(feature_types_config, "cat_text_cardinality_threshold_pct", 0.0) or 0.0
    if pct_threshold > 0.0:
        threshold = min(abs_threshold, max(50, int(total_rows * pct_threshold)))
    else:
        threshold = abs_threshold
    user_assigned = set(text_features) | set(embedding_features)
    promoted: list = []
    cardinalities: dict = {}
    skipped_low_non_null: list = []
    promote_text = feature_types_config.use_text_features
    # honor_user_dtype: pre-cast categorical dtypes (pl.Categorical / pl.Enum / pandas category) are treated
    # as user-declared and skip auto-promotion; raw pl.String / pl.Utf8 / object/string stay candidates.
    honor_user_dtype = getattr(feature_types_config, "honor_user_dtype", False)
    honored_user_dtype_cols: list = []

    if isinstance(df, pl.DataFrame):
        # Accept all embedding-shaped dtypes:
        # - pl.List(pl.Float32/Float64): the legacy variable-length float embedding.
        # - pl.Array(<inner>, N):        polars>=0.20 fixed-size embeddings; backends
        #                                that auto-densify treat the row as a length-N
        #                                vector regardless of inner dtype.
        # - pl.List(pl.Int*):            quantized 8/16/32-bit embeddings (e.g.
        #                                Sentence-Transformers int8 export); the row
        #                                is still a vector, just stored compact.
        _pl_array_cls = getattr(pl, "Array", None)
        _int_inner_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                             pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

        def _is_embedding_dtype(dt) -> bool:
            # Variable-length float embedding (legacy path).
            if dt == pl.List(pl.Float32) or dt == pl.List(pl.Float64):
                return True
            # Quantized int embedding stored as variable-length list.
            inner = getattr(dt, "inner", None)
            if isinstance(dt, pl.List) and inner is not None and inner in _int_inner_dtypes:
                return True
            # Fixed-size pl.Array(...) - any numeric inner is an embedding.
            if _pl_array_cls is not None and isinstance(dt, _pl_array_cls):
                if inner is not None and (inner in (pl.Float32, pl.Float64) or inner in _int_inner_dtypes):
                    return True
            return False

        for name, dtype in df.schema.items():
            if name in user_assigned:
                continue
            if _is_embedding_dtype(dtype):
                if name not in cat_features:
                    embedding_features.append(name)
                continue
            # pl.Enum is an instance-level dtype (not a class), so isinstance() is required alongside the class-level check.
            is_text_like = (
                dtype in (pl.String, pl.Utf8, pl.Categorical)
                or isinstance(dtype, pl.Enum)
            )
            is_user_categorical_dtype = (
                dtype == pl.Categorical or isinstance(dtype, pl.Enum)
            )
            if honor_user_dtype and is_user_categorical_dtype:
                honored_user_dtype_cols.append(name)
                continue
            if is_text_like:
                n_unique = df[name].n_unique()
                if n_unique > threshold:
                    non_null = int(df[name].count())
                    if non_null < min_non_null_abs:
                        skipped_low_non_null.append((name, n_unique, non_null))
                        continue
                    cardinalities[name] = n_unique
                    if promote_text:
                        text_features.append(name)
                        if name in cat_features:
                            promoted.append(name)
                    else:
                        auto_detected_high_card_to_drop.append(name)
    else:
        # pandas: no reliable embedding auto-detect, only text
        for col in df.columns:
            if col in user_assigned:
                continue
            dtype_name = str(df[col].dtype)
            if honor_user_dtype and dtype_name == "category":
                honored_user_dtype_cols.append(col)
                continue
            if dtype_name.startswith("object") or dtype_name.startswith("string") or dtype_name == "category":
                # Skip object columns whose cells are ndarray / list (embedding
                # vectors). nunique() hashes the cells via PyObjectHashTable
                # which raises ``TypeError: unhashable type: 'numpy.ndarray'``.
                # Treat them as embeddings: route to embedding_features and
                # skip the cardinality check (iter#44 fuzz finding).
                if dtype_name.startswith("object"):
                    _series = df[col]
                    try:
                        _first = next((v for v in _series.head(8) if v is not None), None)
                    except Exception:
                        _first = None
                    if _first is not None and (
                        hasattr(_first, "shape")
                        or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
                    ):
                        embedding_features.append(col)
                        if col in cat_features:
                            promoted.append(col)
                        continue
                n_unique = df[col].nunique()
                if n_unique > threshold:
                    non_null = int(df[col].notna().sum())
                    if non_null < min_non_null_abs:
                        skipped_low_non_null.append((col, n_unique, non_null))
                        continue
                    cardinalities[col] = n_unique
                    if promote_text:
                        text_features.append(col)
                        if col in cat_features:
                            promoted.append(col)
                    else:
                        auto_detected_high_card_to_drop.append(col)

    def _fmt_with_cardinality(names):
        parts = []
        for n in names:
            nu = cardinalities.get(n)
            parts.append(f"{n}:{nu:_}" if nu is not None else n)
        return "[" + ", ".join(parts) + "]"

    if verbose and (text_features or embedding_features or promoted):
        if promoted:
            logger.info(
                "  Promoted %d high-cardinality column(s) from cat_features to text_features "
                "(threshold>%s): %s",
                len(promoted), threshold, _fmt_with_cardinality(promoted),
            )
        logger.info(
            "  Auto-detected feature types -- text: %s, embedding: %s",
            _fmt_with_cardinality(text_features) if text_features else "(none)",
            embedding_features or "(none)",
        )

    # Load-bearing: log drop-list regardless of verbose so operators see auto-dropped columns and why (silent drop bites).
    if auto_detected_high_card_to_drop:
        logger.warning(
            "  use_text_features=False: auto-dropping %d high-cardinality "
            "text-like column(s) (n_unique > %d) to prevent "
            "XGB QuantileDMatrix OOM / CB model-artefact bloat: %s. "
            "To keep these columns, set use_text_features=True (routes "
            "them to text_features -- CB uses them, XGB/LGB drop them) "
            "or add them explicitly to feature_types_config.text_features.",
            len(auto_detected_high_card_to_drop),
            threshold,
            _fmt_with_cardinality(auto_detected_high_card_to_drop),
        )

    # Load-bearing diagnostic: columns silently kept as cat_features instead of being promoted (avoids "Dictionary size is 0").
    if skipped_low_non_null:
        formatted = ", ".join(
            f"{name}:{n_unique:_} (non_null={nn:_}/{total_rows:_})"
            for name, n_unique, nn in skipped_low_non_null
        )
        logger.warning(
            "  Auto-detection: %d column(s) had n_unique>%d (would be "
            "promoted to text_features) but non_null<%d (%.1f%% of %d rows, "
            "below the %.2f%% floor) -- kept as cat_features to avoid "
            "CatBoost's 'Dictionary size is 0' error on sparse text "
            "columns: %s",
            len(skipped_low_non_null), threshold, min_non_null_abs,
            min_non_null_frac * 100, total_rows, min_non_null_frac * 100,
            formatted,
        )
    if honored_user_dtype_cols and verbose:
        logger.info(
            "  honor_user_dtype=True: %d column(s) with explicit categorical "
            "dtype (pl.Categorical / pl.Enum / pandas category) kept out of "
            "text-auto-promotion regardless of cardinality: %s",
            len(honored_user_dtype_cols), sorted(honored_user_dtype_cols),
        )

    return text_features, embedding_features, auto_detected_high_card_to_drop


def _validate_feature_type_exclusivity(
    text_features: list,
    embedding_features: list,
    cat_features: list,
) -> None:
    """Raise ValueError if any column appears in multiple feature type lists. Each argument may be ``None`` (treated as empty)."""
    text_features = text_features or []
    embedding_features = embedding_features or []
    cat_features = cat_features or []
    overlap_tc = set(text_features) & set(cat_features)
    if overlap_tc:
        raise ValueError(f"Columns cannot be both text_features and cat_features: {overlap_tc}")
    overlap_ec = set(embedding_features) & set(cat_features)
    if overlap_ec:
        raise ValueError(f"Columns cannot be both embedding_features and cat_features: {overlap_ec}")
    overlap_te = set(text_features) & set(embedding_features)
    if overlap_te:
        raise ValueError(f"Columns cannot be both text_features and embedding_features: {overlap_te}")


def _build_tier_dfs(
    base_dfs: dict,
    strategy,
    text_features: list,
    embedding_features: list,
    tier_cache: dict,
    verbose: bool = False,
) -> dict:
    """Get or create tier-specific DataFrames with unsupported columns removed; returns dict with train/val/test_df trimmed for tier."""
    import polars as pl

    # Cache key must include container-kind: otherwise Polars/pandas tier-DFs collide in a multi-model suite (Linear stashes
    # pandas under (False,False); XGB later gets pandas back and prepare_polars_dataframe raises 'no attribute schema').
    kind = "none"
    for k in ("train_df", "val_df", "test_df"):
        v = base_dfs.get(k)
        if v is not None:
            kind = "pl" if isinstance(v, pl.DataFrame) else "pd"
            break
    tier_key = (strategy.feature_tier(), kind)
    tier = tier_key
    if tier_key in tier_cache:
        return tier_cache[tier_key]

    cols_to_exclude = set()
    if text_features and not strategy.supports_text_features:
        cols_to_exclude.update(text_features)
    if embedding_features and not strategy.supports_embedding_features:
        cols_to_exclude.update(embedding_features)

    if not cols_to_exclude:
        tier_dfs = base_dfs
    else:
        if verbose:
            logger.info("  Tier %s: dropping %d text/embedding columns: %s", tier, len(cols_to_exclude), sorted(cols_to_exclude))
        tier_dfs = {}
        for key in ("train_df", "val_df", "test_df"):
            df_ = base_dfs.get(key)
            if df_ is None:
                tier_dfs[key] = None
            else:
                existing = [c for c in cols_to_exclude if c in df_.columns]
                if not existing:
                    tier_dfs[key] = df_
                elif isinstance(df_, pd.DataFrame):
                    tier_dfs[key] = df_.drop(columns=existing)
                else:
                    # Polars: positional column names (no `columns=` kwarg)
                    tier_dfs[key] = df_.drop(existing)

    tier_cache[tier_key] = tier_dfs
    return tier_dfs


def _split_preds_probs(arr):
    """Regression: 1-D preds; classification: 2-D probs + derived 1-D preds via argmax."""
    if arr is None:
        return None, None
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.argmax(a, axis=1), a
    return a, None


def _maybe_clear_shim_cache(est):
    """Clear XGB/LGB shim caches on estimator if present. Duck-typed via callable check."""
    fn = getattr(est, "clear_cache", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _cfg_get(cfg, key, default=None):
    """Pull ``key`` from a Pydantic-or-dict-or-None config object with a uniform default."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _compute_neural_max_time(non_neural_train_times):
    """Build a Lightning ``trainer_params['max_time']`` dict from P95 of prior non-neural train wall times.

    Returns ``(max_time_dict, p95_seconds, n_samples)`` or ``None`` if no data. Floor at 300 s so a sub-minute
    booster P95 doesn't round to 0h0m and abort Lightning immediately. ``None``, ``[]``, and zero-length ndarrays
    all return ``None``; truthiness of a multi-element ndarray is ambiguous so an explicit length check is needed.
    """
    if non_neural_train_times is None or len(non_neural_train_times) == 0:
        return None
    p95 = float(np.percentile(non_neural_train_times, 95))
    total = max(int(round(p95)), 300)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return (
        {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds},
        p95,
        len(non_neural_train_times),
    )


def _prep_polars_df(_df, strategy, cat_features, category_map):
    """Strategy-driven polars preparation wrapper that lives here (not in core.main) so the
    per-target trainer can import it without re-entering the main module - main imports
    _train_one_target, so referencing _prep_polars_df from there would form an import cycle."""
    if _df is None:
        return None
    if category_map is not None:
        return strategy.prepare_polars_dataframe(_df, cat_features, category_map=category_map)
    return strategy.prepare_polars_dataframe(_df, cat_features)


def _bulk_setattr_to_ctx(ctx, names: tuple[str, ...], values: dict) -> None:
    """Bulk-assign each name from ``values`` onto ``ctx``. Raises KeyError on missing name.

    Used by core/main.py to mirror local variables onto the suite TrainingContext during
    the phase->ctx migration. Fails loudly when a slot name is missing from ``values`` so
    partial-migration bugs (like the prior ``train_df_pandas_pre`` slot miss) surface at
    call time instead of as ``AttributeError: 'NoneType' has no attribute 'foo'`` later.
    """
    missing = [n for n in names if n not in values]
    if missing:
        raise KeyError(f"_bulk_setattr_to_ctx: names missing from values dict: {missing}")
    for n in names:
        setattr(ctx, n, values[n])
