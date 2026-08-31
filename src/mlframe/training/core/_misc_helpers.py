"""Small utility functions: logging, metrics, DataFrame ops, validation, Polars helpers, dataset reuse detection, tier-DF building."""

from __future__ import annotations

import logging
import os
import sys
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..utils import filter_existing, compute_model_input_fingerprint, _dtype_family
from mlframe.utils.log_throttle import log_throttle

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
        def _is_timestamped(h) -> bool:
            """True if handler ``h``'s formatter already includes ``%(asctime)``."""
            existing = getattr(h.formatter, "_fmt", None) if h.formatter else None
            return bool(existing and "%(asctime)" in existing)

        # ALL handlers must already be timestamped, not just the first one found -- a handler appended
        # by another package (e.g. Jupyter) BETWEEN two calls would otherwise never get upgraded, since
        # an earlier-installed, already-fixed handler iterated first would trigger this early return.
        if all(_is_timestamped(h) for h in root.handlers):
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
    """Pull a per-split per-name metric value, tolerating nested/flat/split-less/class-indexed shapes; NaN on a miss.

    Two shapes were missing and both produced the same silent symptom -- a cross-target verdict table whose
    ``best_model`` column read ``-`` while the log printed that model's metrics a few lines above:

    - a CLASS-INDEXED classification layout, ``metrics[split][1][name]``, which is what every binary and
      multiclass run produces (the sibling reader in ``_ensemble_chooser`` already drills this level);
    - an entry that is still the raw ``(namespace, train_df, val_df, test_df)`` tuple rather than the namespace.
    """
    if isinstance(entry, tuple) and entry:
        entry = entry[0]
    metrics = getattr(entry, "metrics", None)
    if not isinstance(metrics, dict):
        return float("nan")
    inner = metrics.get(split)
    if isinstance(inner, dict):
        v = inner.get(name)
        if isinstance(v, (int, float)):
            return float(v)
        # Class-indexed: {1: {...}} for binary, {0: {...}, 1: {...}, ...} for multiclass. The positive class is
        # the reported one for binary; for multiclass take the first class that carries the metric, which
        # matches what the per-class report prints.
        for _cls_key in sorted(k for k in inner if isinstance(k, int)):
            _cls = inner.get(_cls_key)
            if isinstance(_cls, dict):
                _cv = _cls.get(name)
                if isinstance(_cv, (int, float)):
                    return float(_cv)
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
    added: list = []
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
        """Concat the dropped high-cardinality columns back onto ``frame`` as one block (polars: ``with_columns``; pandas: single ``pd.DataFrame`` concat to avoid per-column fragmentation)."""
        if frame is None or not extras:
            return frame
        if isinstance(frame, pl.DataFrame):
            return frame.with_columns([pl.Series(c, v) for c, v in extras.items()])
        # Add all extras as ONE block via concat instead of per-column
        # assign/insert: the latter triggers pandas' "highly fragmented"
        # PerformanceWarning on wide frames (each insert grows the block
        # count). drop colliding names first to preserve assign's
        # overwrite semantics. Fresh frame -> source is not mutated.
        extra_df = pd.DataFrame(extras, index=frame.index)
        dup = [c for c in extra_df.columns if c in frame.columns]
        if dup:
            frame = frame.drop(columns=dup)
        return pd.concat([frame, extra_df], axis=1)

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
            col_vals = _split_df[col_name].to_numpy() if hasattr(_split_df[col_name], "to_numpy") else _np.asarray(_split_df[col_name])
        except Exception:
            logger.debug("failed materialising column %r from split frame; skipping", col_name, exc_info=True)
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
    """Raise ValueError if ``path`` is not inside ``trusted_root``; gates ``joblib.load`` to limit arbitrary-code-execution surface.

    Thin re-export of the single shared implementation (``mlframe.core.helpers.validate_trusted_path``).
    """
    from mlframe.core.helpers import validate_trusted_path as _validate

    _validate(path, trusted_root)


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
        caps["cb_pool_label_swap"] = caps["cb_pool_set_label"] and caps["cb_pool_set_weight"]
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

    Key resolution order (post-fix SKEW-COL-ORDER): prefers the explicit ``metadata["raw_input_columns"]``
    (set by ``_phase_fit_pipeline`` before transform), falls back to legacy ``metadata["input_columns"]``
    (the same content under the old alias), and finally to ``metadata["columns"]`` (post-pipeline; back-
    compat for models trained before the explicit-key fix landed). The raw-input schema is the right
    anchor for predict-time validation: pipelines may rename/add columns (one-hot expansion, dim_reducer
    output, TF-IDF), so validating against post-pipeline names drops every raw user column as "extra".
    """
    columns = metadata.get("raw_input_columns") or metadata.get("input_columns") or metadata.get("columns", [])
    if not columns:
        return df

    # Augment the allowlist with columns produced by the suite-owned
    # datetime decomposition (metadata["datetime_methods"] = {src_col:
    # {method: dtype_name, ...}}). The raw-input snapshot is taken BEFORE
    # the suite runs ``create_date_features``, so the derived
    # ``<src>_<method>`` columns look "extra" to a strict set-diff and
    # get dropped here -- which then breaks the trained pipeline that
    # expects them. The replay step has already added them to ``df``
    # by the time we reach this validator; treating them as allowed
    # is the correct invariant. FTE-owned datetime expansions
    # (``ftextractor_emitted_columns``) similarly need to count.
    _allowed = list(columns)
    _dt_methods_map = metadata.get("datetime_methods") or {}
    # ``create_date_features`` at training time also emits cyclical companions
    # for each source column when ``add_cyclical=True`` (its default; the suite-side
    # fit-pipeline call relies on the default). The metadata only records the
    # configured integer accessors (year/month/day/...), so the allowlist must
    # additionally include ``<src>_<period>_sin`` / ``<src>_<period>_cos`` for
    # each period in ``_DEFAULT_CYCLICAL_PERIODS`` so the validator does not
    # mis-classify them as "extra" and drop them. Predict-time replay also runs
    # with default ``add_cyclical=True`` so the columns are present in the frame.
    from mlframe.feature_engineering.basic import _DEFAULT_CYCLICAL_PERIODS
    _cyclical_period_names = [_p for _p, _ in _DEFAULT_CYCLICAL_PERIODS]
    for _src, _methods in _dt_methods_map.items():
        _allowed.extend(f"{_src}_{_method}" for _method in _methods or {})
        for _period in _cyclical_period_names:
            _allowed.append(f"{_src}_{_period}_sin")
            _allowed.append(f"{_src}_{_period}_cos")
    _fte_emitted = metadata.get("ftextractor_emitted_columns") or {}
    for _emitted_list in _fte_emitted.values():
        if isinstance(_emitted_list, (list, tuple, set)):
            _allowed.extend(_emitted_list)
    # Dedupe while preserving order. ``raw_input_columns`` / ``input_columns``
    # already contain the FTE-emitted + suite-decomposed derived columns
    # (they snapshot the post-pipeline schema); appending the same names
    # again from ``datetime_methods`` / ``ftextractor_emitted_columns``
    # produced duplicate entries. Downstream ``df[filter_existing(df,
    # columns)]`` then selected the SAME column twice and polars rejected
    # the result with ``DuplicateError: column with name 'ts_month' has
    # more than one occurrence``. dict.fromkeys preserves first-seen
    # order (Python 3.7+) which matches the snapshot's column layout.
    columns = list(dict.fromkeys(_allowed))

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
            "Missing columns in input: %s. The pipeline will attempt " "to proceed -- downstream errors about shape mismatches " "usually trace back here.",
            sorted(missing_cols),
        )

    # Canonicalise column ORDER to the trained schema, not just on the extra-columns path. sklearn-API
    # estimators raise on a same-names-different-order frame (they validate feature_names_in_ order, never
    # silently reorder), and positional consumers (raw boosters, numpy ``.values`` paths) would mis-map a
    # reordered frame. ``df[filter_existing(...)]`` is a name-based view-select on both flavours (no whole-
    # frame copy), so reordering a benignly-permuted serving frame is cheap and prevents an all-models-fail.
    kept_in_schema_order = filter_existing(df, columns)
    if extra_cols and verbose:
        logger.info("Dropping extra columns: %s", sorted(extra_cols))
    if extra_cols or list(df.columns) != kept_in_schema_order:
        df = df[kept_in_schema_order]

    # Per-model input-schema diff: HARD-FAIL on changes that silently corrupt predictions
    # (removed cat/text/embedding cols, role changes, dtype family changes for non-numeric roles);
    # SOFT-WARN on benign differences the pipeline casts transparently (float32<->float64, etc.).
    # Silent pass on old metadata files predating model_schemas.
    model_schemas = metadata.get("model_schemas")
    if model_schemas:
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
                            family_changes.append(f"    {col}: trained={e['dtype']!r} ({ef}) serving={live['dtype']!r} ({lf})")
                        else:
                            soft_family_changes.append(f"    {col}: trained={e['dtype']!r} ({ef}) serving={live['dtype']!r} ({lf}) (numeric role)")
                    else:
                        soft_width_changes.append(f"    {col}: trained={e['dtype']!r} serving={live['dtype']!r} (same family={lf})")
            hard_fail = bool(critical_removed or family_changes or role_changes)
            if hard_fail:
                diff_lines = []
                if critical_removed:
                    diff_lines.append(f"  - critical missing (cat/text/embedding): {sorted(critical_removed)}")
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
                    f"(trained hash={expected_hash}, serving hash={live_hash}):\n" + "\n".join(diff_lines) + "\n"
                    "Either restore the upstream feature pipeline that produced "
                    "the trained-time layout, or retrain the model against the "
                    "current serving frame."
                )
            if soft_width_changes or soft_family_changes:
                lines: list = []
                if soft_width_changes:
                    lines.extend(s.strip() for s in soft_width_changes)
                if soft_family_changes:
                    lines.extend(s.strip() for s in soft_family_changes)
                log_throttle(
                    logger,
                    "misc_helpers_input_schema_drift",
                    logging.WARNING,
                    "Input-schema drift for %s (pipeline-internal casts on "
                    "numeric-role columns and/or width-only changes). "
                    "Accepting; trained pipeline is responsible for "
                    "casting the serving df: %s",
                    model_file_name,
                    "; ".join(lines),
                )

    return df


# Feature-type detection lives in a sibling: this file crossed the 1000-LOC budget. Re-exported so existing
# ``from ._misc_helpers import _auto_detect_feature_types`` imports keep resolving.
from ._misc_helpers_feature_types import (  # noqa: F401
    _auto_detect_feature_types,
    _filter_polars_cat_features_by_dtype,
    _validate_feature_type_exclusivity,
)


def _build_tier_dfs(
    base_dfs: dict,
    strategy,
    text_features: list,
    embedding_features: list,
    tier_cache: dict[Any, dict],
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
            logger.debug("clear_cache() raised on estimator %r; ignoring", type(est).__name__, exc_info=True)
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
    total = max(round(p95), 300)
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


_CTX_STRICT = os.environ.get("MLFRAME_CTX_STRICT", "").strip().lower() in ("1", "true", "yes", "on")


def _bulk_setattr_to_ctx(ctx, names: tuple[str, ...], values: dict) -> None:
    """Bulk-assign each name from ``values`` onto ``ctx``. Raises KeyError on missing name.

    Used by core/main.py to mirror local variables onto the suite TrainingContext during
    the phase->ctx migration. Fails loudly when a slot name is missing from ``values`` so
    partial-migration bugs (like the prior ``train_df_pandas_pre`` slot miss) surface at
    call time instead of as ``AttributeError: 'NoneType' has no attribute 'foo'`` later.

    Under ``MLFRAME_CTX_STRICT=1`` each migrated slot is identity-checked against its local
    after the copy, so a future slot-miss (slot name typo, stale dataclass) surfaces here
    rather than as a silent stale-value downstream. Identity-only (no value compare / hash)
    keeps the check safe on 100+GB frames.
    """
    missing = [n for n in names if n not in values]
    if missing:
        raise KeyError(f"_bulk_setattr_to_ctx: names missing from values dict: {missing}")
    for n in names:
        setattr(ctx, n, values[n])
    if _CTX_STRICT:
        mismatched = [n for n in names if getattr(ctx, n) is not values[n]]
        if mismatched:
            raise AssertionError(f"_bulk_setattr_to_ctx: ctx slot(s) diverged from locals after copy: {mismatched}")

def mirror_split_outputs_to_ctx(ctx: Any, **values: Any) -> None:
    """Mirror the post-split locals the outlier-detection phase reads onto ``ctx``.

    Takes them by keyword rather than a ``locals()`` snapshot: the slot names and the values then live side by
    side, so a renamed local fails loudly at the call instead of silently ceasing to be mirrored.
    """
    _bulk_setattr_to_ctx(ctx, tuple(values), values)
