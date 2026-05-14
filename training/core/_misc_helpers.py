"""Small utility functions extracted from ``core/utils.py``.

Logging, metrics, DataFrame operations, validation, Polars helpers,
dataset reuse detection, and tier-DF building.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from ..utils import filter_existing

logger = logging.getLogger(__name__)

def _ensure_logging_visible(level: int = logging.INFO) -> None:
    """Make mlframe progress logs visible -- with timestamps -- in Jupyter and
    plain scripts.

    Two cases:
      1. No root handlers at all (bare Python / fresh Jupyter kernel before
         anyone called `logging.basicConfig`). -> install a stdout handler
         with ``%(asctime)s %(levelname)s %(name)s: %(message)s`` format.
      2. Root already has handlers but their formatter lacks a timestamp
         (classic Jupyter/IPython default emits just ``LEVEL:name:message``,
         which is useless for profiling long training runs). -> replace
         those formatters in place with the timestamped one. Handlers that
         already format with ``%(asctime)s`` are left untouched so a user
         who intentionally configured a custom format isn't clobbered.
    """
    root = logging.getLogger()
    desired_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    desired_datefmt = "%H:%M:%S"
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
    """Pull a per-split per-name metric value out of an entry.

    Entries shipped by the per-target loop carry a ``metrics`` mapping
    in any of several shapes (legacy), e.g.:
    - ``entry.metrics[split][name]`` -> {'train': {'RMSE': 0.42, ...}}
    - ``entry.metrics[name]`` -> {'RMSE': 0.42, ...} (split-less)
    - flat key like ``f"{split}_{name}"``: ``{'train_RMSE': 0.42}``

    Returns ``float('nan')`` on any miss so callers can treat absent
    metrics uniformly.
    """
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
    """Re-attach pre-drop high-card cat columns to ``train/val/test_df``.

    Used by the dummy_baselines per-target call site so per_group_mean
    can use group keys (e.g. ``well_id`` with 600+ unique values) that
    were stripped from tree-model frames to prevent XGB QuantileDMatrix
    OOM. Captured pre-drop ndarrays are sliced by ``train_od_idx`` /
    ``val_od_idx`` so the re-added column row-aligns to the OD-filtered
    frame; test is never OD-filtered. Returns
    ``(train_df, val_df, test_df, added_col_names)``.
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
    """Reassemble a single column at the FULL n_total row index space
    from the per-split frames produced upstream of the per-target loop.

    Used by composite-target discovery integration: discovery's
    ``forward`` transform needs the base column at every row in the
    full df (so val and test rows get T values for the per-target
    training loop's slicing). The base column is split across
    ``train_df`` / ``val_df`` / ``test_df`` after the suite's
    train/val/test partition; this helper writes each split's column
    back into a single n_total-sized ndarray indexed by the original
    split indices.

    Parameters
    ----------
    col_name : str
        Column to extract.
    train_df, val_df, test_df : pandas.DataFrame | polars.DataFrame | None
        Per-split frames. ``val_df`` / ``test_df`` may be None when
        the suite was configured without a val or test split.
    train_idx, val_idx, test_idx : ndarray | None
        Row indices INTO the n_total-sized full frame.
    n_total : int
        Size of the full row index space.

    Returns
    -------
    ndarray
        Float64 array of length ``n_total`` with the column values
        slotted at the appropriate indices. Rows not covered by any
        split (rare, but possible when the FTE drops rows) keep NaN.
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
            # Frame and index disagree (e.g. OD-filtered train_df
            # paired with raw train_idx). Skip rather than
            # mis-aligning silently.
            continue
        out[idx_arr] = col_vals
    return out




def _drop_cols_df(df, cols):
    """Drop ``cols`` from ``df`` (pandas or Polars), ignoring missing names.

    Centralizes the 4-line `isinstance(df, pd.DataFrame)` branch that appeared in
    both `predict_mlframe_models_suite` and `predict_from_models`.
    """
    import pandas as _pd  # local import to avoid top-level cost during helper init
    if not cols:
        return df
    existing = filter_existing(df, cols)
    if not existing:
        return df
    if isinstance(df, _pd.DataFrame):
        return df.drop(columns=existing, errors="ignore")
    return df.drop(existing)  # Polars




def _validate_trusted_path(path: str, trusted_root):
    """Raise ValueError if ``path`` is not inside ``trusted_root``.

    Mirrors the `mlframe.inference.read_trained_models` convention. Gating every
    ``joblib.load`` of a pickled metadata/model file keeps arbitrary-code-execution
    surface limited to explicitly-opted-in directories.
    """
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
        raise ValueError(f"Path {abs_path} is not inside trusted_root {abs_root}")
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




def _detect_dataset_reuse_capabilities() -> "Dict[str, bool]":
    """Feature-detect which GBDT sklearn wrappers can accept a pre-built
    dataset as ``X``, enabling label/weight reuse across fits without
    rebuilding the native data structure.

    Matrix of capability keys:

    - ``cb_pool_set_label``: ``catboost.Pool.set_label`` callable.
    - ``cb_pool_set_weight``: ``catboost.Pool.set_weight`` callable.
    - ``cb_pool_label_swap``: both of the above AND
      ``CatBoostClassifier.fit(X=Pool)`` short-circuits the rebuild
      (verified via ``_build_train_pool`` code path in the installed
      catboost build).
    - ``xgb_dmatrix_set_label`` / ``xgb_dmatrix_set_weight``: ``DMatrix``
      exposes both mutators (true in every 3.x release).
    - ``xgb_sklearn_accepts_dmatrix``: ``XGBClassifier.fit(X=DMatrix)``
      short-circuits -- empirically False in 3.2.0 (upstream FR pending).
    - ``lgb_dataset_set_label`` / ``lgb_dataset_set_weight``: ``Dataset``
      exposes both mutators (true in every 4.x release).
    - ``lgb_sklearn_accepts_dataset``: ``LGBMClassifier.fit(X=Dataset)``
      short-circuits -- empirically False in 4.6.0 (upstream FR pending).

    Only the capability set produced here gates Fix 9.4.3 reuse; the
    upstream-pending items stay False until the libraries ship the
    short-circuit and a user upgrades.
    """
    caps: "Dict[str, bool]" = {}

    # CatBoost
    try:
        import catboost as _cb
        _pool_cls = getattr(_cb, "Pool", None)
        caps["cb_pool_set_label"] = callable(getattr(_pool_cls, "set_label", None))
        caps["cb_pool_set_weight"] = callable(getattr(_pool_cls, "set_weight", None))
        # Short-circuit check: CatBoostClassifier.fit(X=Pool) is supported
        # in every CB >= 1.0 via ``_build_train_pool`` (``isinstance(X,
        # Pool)`` return). The label-swap variant lands with the PR that
        # made Pool.set_label callable -- gate the reuse on BOTH.
        caps["cb_pool_label_swap"] = (
            caps["cb_pool_set_label"] and caps["cb_pool_set_weight"]
        )
    except ImportError:
        caps["cb_pool_set_label"] = False
        caps["cb_pool_set_weight"] = False
        caps["cb_pool_label_swap"] = False

    # XGBoost
    try:
        import xgboost as _xgb
        _dm = getattr(_xgb, "DMatrix", None)
        caps["xgb_dmatrix_set_label"] = callable(getattr(_dm, "set_label", None))
        caps["xgb_dmatrix_set_weight"] = callable(getattr(_dm, "set_weight", None))
        # Upstream wrapper does NOT short-circuit yet (verified 2026-04-21
        # on xgboost 3.2.0 -- ``_create_dmatrix`` rebuilds unconditionally).
        # Mark False until an upstream PR lands.
        caps["xgb_sklearn_accepts_dmatrix"] = False
    except ImportError:
        caps["xgb_dmatrix_set_label"] = False
        caps["xgb_dmatrix_set_weight"] = False
        caps["xgb_sklearn_accepts_dmatrix"] = False

    # LightGBM
    try:
        import lightgbm as _lgb
        _ds = getattr(_lgb, "Dataset", None)
        caps["lgb_dataset_set_label"] = callable(getattr(_ds, "set_label", None))
        caps["lgb_dataset_set_weight"] = callable(getattr(_ds, "set_weight", None))
        # Same story as XGBoost -- verified 2026-04-21 on lightgbm 4.6.0.
        caps["lgb_sklearn_accepts_dataset"] = False
    except ImportError:
        caps["lgb_dataset_set_label"] = False
        caps["lgb_dataset_set_weight"] = False
        caps["lgb_sklearn_accepts_dataset"] = False

    return caps




def _validate_input_columns_against_metadata(
    df,
    metadata: "Dict[str, Any]",
    verbose: bool = False,
):
    """Validate inference-time DataFrame columns against model metadata.

    Before this helper (inline in ``predict_mlframe_models_suite`` /
    ``predict_from_models`` up to 2026-04-19), the logic was:
      - WARN on missing columns, then proceed
      - Drop extra columns if any

    Problem: if a missing column was a load-bearing ``cat_features`` /
    ``text_features`` / ``embedding_features`` member, the pipeline
    transform + model predict ran on a shape-mismatched frame and
    either (a) crashed deep inside sklearn with ``X has N features,
    expected M``, or (b) produced garbage predictions. The WARN alone
    was not actionable.

    Now: columns are partitioned by severity:
      - Missing load-bearing features (cat/text/embedding): **raise
        ValueError** with a diagnostic naming them. These cannot be
        safely dropped -- the pipeline was fitted with them.
      - Other missing columns: WARN + proceed. Some callers drop
        derived columns that the pipeline reconstructs; that's OK.
      - Extra columns: dropped silently (or logged at verbose=True).

    Returns the df (possibly with extra columns filtered out).
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

    # Fix 8f (2026-04-21, v2): per-model input-schema diff reporting.
    # ``metadata['model_schemas']`` (if present) maps model_file_name to
    # ``{schema_hash, input_schema, mlframe_model, weight_name}`` -- the
    # exact realised layout each fitted model saw at training time.
    #
    # Severity rules at load time:
    #   HARD-FAIL (ValueError) on changes sklearn / CB / XGB / LGB will
    #   silently produce wrong predictions for:
    #     * removed columns that were cat/text/embedding features
    #     * role changes (cat -> text, text -> numeric, etc.)
    #     * dtype FAMILY changes (string -> numeric, numeric -> categorical)
    #   SOFT-WARN on benign differences the downstream pipeline casts
    #   transparently:
    #     * float32 <-> float64, int32 <-> int64 (width-only)
    #     * added columns (caller superset -- already filtered to the
    #       trained subset above)
    # Silent pass on old metadata files that predate Fix 8.
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
            # Classify diffs by severity.
            # Trained snapshot is POST-pipeline (what fit() actually saw);
            # live snapshot is PRE-pipeline (raw serving df). For cat / text /
            # embedding columns the dtype/role is user-declared and stable
            # across train<->serve, so family changes there ARE critical
            # (silent drift in label encoding / tokenizer vocab / etc.). For
            # numeric-role columns the pipeline internally casts and encodes
            # (OHE of object, label-encoding, float32 downcast, etc.), so
            # family changes are expected and must be soft-warned.
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
                l = live_schema_idx[col]
                role_critical = e["role"] in ("cat", "text", "embedding") or l["role"] in ("cat", "text", "embedding")
                if l["role"] != e["role"]:
                    if role_critical:
                        role_changes.append(f"    {col}: trained role={e['role']} serving role={l['role']}")
                if l["dtype"] != e["dtype"]:
                    ef = _dtype_family(e["dtype"])
                    lf = _dtype_family(l["dtype"])
                    if ef != lf:
                        if role_critical:
                            family_changes.append(
                                f"    {col}: trained={e['dtype']!r} ({ef}) serving={l['dtype']!r} ({lf})"
                            )
                        else:
                            soft_family_changes.append(
                                f"    {col}: trained={e['dtype']!r} ({ef}) serving={l['dtype']!r} ({lf}) (numeric role)"
                            )
                    else:
                        soft_width_changes.append(
                            f"    {col}: trained={e['dtype']!r} serving={l['dtype']!r} (same family={lf})"
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
    df: "pl.DataFrame",
    cat_features: "List[str]",
) -> "List[str]":
    """Defensive filter for CB Polars fastpath ``cat_features``.

    CatBoost 1.2.x's ``_set_features_order_data_polars_categorical_column``
    is a Cython fused cpdef with dispatch **only** for ``pl.Categorical``
    (and on some builds ``pl.Enum``). If a caller hands a column to CB
    via ``cat_features`` but the column's dtype in the DataFrame is
    ``pl.String``/``pl.Utf8``/numeric/etc, the dispatcher falls through
    to the opaque ``TypeError: No matching signature found`` -- with no
    hint about which column or why.

    Production incident 2026-04-19: the orchestration in
    ``train_mlframe_models_suite`` short-circuited to a *stale* pre-promotion
    cat_features list that still contained 4 columns which had been
    cast ``pl.Categorical -> pl.String`` for the text-features fastpath.
    CB saw ``cat_features=['category', 'skills_text', ...]`` with those
    columns being ``pl.String`` and raised "No matching signature found",
    burning 22 s + a 150 s pandas fallback on every run.

    This helper runs **right before** passing cat_features to
    ``model.fit()``:
      - keeps columns whose dtype is ``pl.Categorical`` or ``pl.Enum``
      - drops columns with any other dtype and logs a WARNING naming
        them and their observed dtype
      - silently drops columns missing from the DataFrame (defensive;
        a missing column would crash CB with a different error anyway)

    Returns the filtered list; empty list if nothing valid remains.
    """
    valid: list = []
    dropped: list = []  # list of (col_name, dtype_str)
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
    """Auto-detect text and embedding features from DataFrame schema and cardinality.

    Also *promotes* columns that were initially classified as categorical by the
    Polars schema (e.g. columns ending in ``_text`` that are ``pl.Categorical``
    in raw data) up to ``text_features`` when their cardinality exceeds the
    configured threshold. Previously such columns stayed in ``cat_features``
    and CatBoost wasted GB of memory on nominal encoding of what are really
    free-text blobs. The promotion happens iff:

        * user did not explicitly list the column in ``text_features``
          or ``embedding_features`` already, AND
        * the column's dtype is ``pl.String``/``pl.Utf8``/``pl.Categorical``
          (ordered categoricals like ``pl.Enum`` stay nominal), AND
        * ``n_unique > cat_text_cardinality_threshold``.

    Promoted columns are returned in ``text_features``; the caller is
    responsible for filtering its own ``cat_features`` against the returned
    ``text_features`` (see ``effective_cat_features`` construction in
    ``train_mlframe_models_suite``). This function does NOT mutate the
    ``cat_features`` argument -- prior versions did, which created a latent
    repeat-call state-leak trap whenever a caller reused the same list.

    Args:
        df: Training DataFrame (Polars or pandas).
        feature_types_config: FeatureTypesConfig with user overrides and threshold.
        cat_features: Already-detected categorical features (from pipeline).
            Read-only. Used only to decide which auto-detected text columns
            were "promoted" from cat_features vs newly discovered.
        verbose: Whether to log detections.

    Returns:
        (text_features, embedding_features) -- lists of column names.
    """
    import polars as pl

    _ftc = feature_types_config
    text_features = list(_ftc.text_features or []) if _ftc is not None else []
    embedding_features = list(_ftc.embedding_features or []) if _ftc is not None else []
    # Auto-detected high-cardinality text-like columns that the caller
    # should DROP entirely from the training df when ``use_text_features=
    # False``. Semantic:
    #   ``use_text_features=True``  -> auto-detected cols go into
    #     ``text_features`` (CB uses them; XGB/LGB drop them via
    #     ``supports_text_features=False`` mechanism).
    #   ``use_text_features=False`` -> auto-detected cols go into THIS
    #     list so the caller can drop them from the df entirely (so
    #     no model -- including CB -- tries to consume them as a 2M-level
    #     categorical, which otherwise OOMs XGB's QuantileDMatrix and
    #     balloons CB's model artefact).
    # Regardless of the flag, if the user explicitly listed a column in
    # ``feature_types_config.text_features``/``embedding_features``, that
    # column is honored (not touched here).
    auto_detected_high_card_to_drop: list = []

    # Master switch ``use_text_features`` only gates AUTO-PROMOTION (the
    # cardinality-based heuristic below). User-supplied explicit
    # ``text_features`` list is honored regardless -- if the user passed it,
    # they intend those columns routed as text_features. (2026-04-21
    # refinement: earlier version cleared the explicit list too, which
    # broke ``test_non_catboost_drops_text_columns`` / auto-detection
    # tests that pass an explicit list AND expect no auto-promotion on
    # top. The ``promote_text`` flag below is the real gate.)

    if feature_types_config is None or not feature_types_config.auto_detect_feature_types:
        return text_features, embedding_features, auto_detected_high_card_to_drop

    # Defensive: callers sometimes pass ``cat_features=None`` (e.g. after a
    # model skipped categorical detection). Treat as empty list -- the
    # ``if name in cat_features`` checks below would otherwise crash with
    # ``TypeError: argument of type 'NoneType' is not iterable``.
    if cat_features is None:
        cat_features = []

    threshold = feature_types_config.cat_text_cardinality_threshold
    # Minimum non-null FRACTION required to promote a string/categorical
    # column to text_features. CatBoost's text feature estimator builds
    # a TF-IDF vocabulary from the column's non-null content; with too
    # few non-null samples the ``occurrence_lower_bound`` filter prunes
    # everything and the estimator raises
    #   ``catboost/.../text_feature_estimators.cpp:89:
    #     Dictionary size is 0, check out data or try to decrease
    #     occurrence_lower_bound parameter``
    # (observed 2026-04-19 on ``_raw_countries`` and ``job_post_source``
    # in prod -- both ``n_unique > 50`` but >99.9% null, yielding a
    # handful of non-null strings total and an empty dictionary after
    # occurrence filtering).
    #
    # Using a FRACTION (not absolute count) so the guard scales with
    # dataset size: a 50-row test DF with 50 non-null rows (fraction 1.0)
    # passes, while an 810k-row prod DF with 6 non-null (fraction 7e-6)
    # fails. Default 0.01 = 1%: anything below that in a typical
    # many-hundred-row+ frame is a sparse column that CB's TF-IDF
    # cannot build a vocabulary from.
    min_non_null_frac = getattr(
        feature_types_config, "min_non_null_fraction_for_text_promotion", 0.01
    )
    # Total row count -- denominator for the fraction. For pandas this
    # is len(df); for polars, df.height.
    total_rows = df.height if hasattr(df, "height") else len(df)
    # Translate the fraction back to an absolute count for the guard --
    # avoids per-column float division inside the loop and reuses the
    # same floor for every column on this DF.
    min_non_null_abs = max(1, int(round(total_rows * min_non_null_frac)))
    user_assigned = set(text_features) | set(embedding_features)
    promoted: list = []  # cat_features -> text_features, tracked for diagnostic log only
    cardinalities: dict = {}  # per auto-detected text col: n_unique (for diagnostic log)
    skipped_low_non_null: list = []  # (name, n_unique, non_null_count) -- blocked by guard
    # Master-switch short-circuits the text-promotion branches (embedding
    # detection still runs). Cheaper than threading the flag through every
    # append site; one flag read per schema iteration at worst.
    promote_text = feature_types_config.use_text_features
    # 2026-04-21 ``honor_user_dtype``: when True, pre-cast categorical
    # dtypes (pl.Categorical / pl.Enum / pandas ``category``) are treated
    # as user-declared and exempt from auto-promotion. Raw pl.String /
    # pl.Utf8 / pandas object/string columns remain promotion candidates.
    honor_user_dtype = getattr(feature_types_config, "honor_user_dtype", False)
    honored_user_dtype_cols: list = []  # for diagnostic log

    if isinstance(df, pl.DataFrame):
        for name, dtype in df.schema.items():
            if name in user_assigned:
                continue
            # Embedding: pl.List(pl.Float32/Float64)
            if dtype == pl.List(pl.Float32) or dtype == pl.List(pl.Float64):
                if name not in cat_features:
                    embedding_features.append(name)
                continue
            # String/Categorical/Enum -- evaluate cardinality to split cat vs text.
            # pl.Enum is a fixed-domain categorical; it has an instance-level
            # dtype object (not a class), so it doesn't compare equal to the
            # class-level check above. Use isinstance() for Enum specifically.
            is_text_like = (
                dtype in (pl.String, pl.Utf8, pl.Categorical)
                or isinstance(dtype, pl.Enum)
            )
            # honor_user_dtype: skip promotion for already-categorical
            # dtypes; only raw Utf8/String remain auto-promotion candidates.
            is_user_categorical_dtype = (
                dtype == pl.Categorical or isinstance(dtype, pl.Enum)
            )
            if honor_user_dtype and is_user_categorical_dtype:
                honored_user_dtype_cols.append(name)
                continue
            if is_text_like:
                n_unique = df[name].n_unique()
                if n_unique > threshold:
                    # Non-null FRACTION guard -- block promotion/drop if
                    # the column is sparse relative to total rows. For
                    # the PROMOTE path this keeps CB's text estimator
                    # from producing an empty TF-IDF dictionary; for the
                    # DROP path (``use_text_features=False``) the
                    # sparseness check is still a useful signal that
                    # the column is unlikely to materially help any
                    # model anyway -- callers handle it identically.
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
                        # ``use_text_features=False``: caller MUST drop
                        # this column. Leaving it as cat_feature crashes
                        # XGB (QuantileDMatrix OOM on 2M-level cats) and
                        # balloons CB artefact size with a useless
                        # nominal-encoding vocabulary.
                        auto_detected_high_card_to_drop.append(name)
    else:
        # pandas: only detect high-cardinality text (no reliable embedding auto-detect)
        for col in df.columns:
            if col in user_assigned:
                continue
            dtype_name = str(df[col].dtype)
            # honor_user_dtype symmetry with the polars branch: ``category``
            # dtype is a user-declared categorical in pandas land; skip
            # promotion when the flag is on. ``object`` / ``string`` stay
            # as auto-promotion candidates.
            if honor_user_dtype and dtype_name == "category":
                honored_user_dtype_cols.append(col)
                continue
            if dtype_name.startswith("object") or dtype_name.startswith("string") or dtype_name == "category":
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
                        # ``use_text_features=False``: drop-list for caller.
                        auto_detected_high_card_to_drop.append(col)

    # Historical note: this function used to mutate ``cat_features`` in place
    # (calling ``.remove(name)`` for each promoted column). The in-place removal
    # was redundant -- the actual caller filter lives at the call site, where
    # ``effective_cat_features`` is built via set-difference against
    # ``text_features``. We removed the mutation so repeat calls with a shared
    # list don't corrupt the caller's state.

    def _fmt_with_cardinality(names):
        """'col1:500, col2:12_345' -- makes it obvious *why* a column was
        promoted vs the configured threshold."""
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

    # Load-bearing: log the drop-list regardless of verbose. Operator
    # needs to see WHICH columns were auto-dropped and WHY -- a silent
    # drop is exactly the class of bug we just fixed (2026-04-22):
    # skills_text at 2M unique silently stayed as cat_feature under
    # ``use_text_features=False`` and OOM'd XGB on a prod 9M-row run.
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

    # Always log the skipped-by-non-null-guard set, even at verbose=False
    # -- this is a load-bearing diagnostic: columns that would otherwise
    # have been promoted and crashed CatBoost with "Dictionary size is 0"
    # are silently kept as cat_features. The operator needs to know so
    # they can either (a) fix the upstream feature-extraction to produce
    # more non-null samples, or (b) accept the lower-quality cat usage
    # of these columns.
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
    """Raise ValueError if any column appears in multiple feature type lists.

    Each argument may be ``None`` (treated as empty) -- callers that skipped
    one of the feature-type detection stages (e.g. models without
    categorical awareness) pass None rather than ``[]``.
    """
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
    """Get or create tier-specific DataFrames with unsupported columns removed.

    Uses .select() instead of .drop() to avoid unnecessary full-DF copies.

    Args:
        base_dfs: Dict with keys train_df, val_df, test_df.
        strategy: ModelPipelineStrategy for the current model.
        text_features: Text feature column names.
        embedding_features: Embedding feature column names.
        tier_cache: Mutable dict caching tier DFs (tier_key -> dict of DFs).
        verbose: Log column dropping.

    Returns:
        Dict with train_df, val_df, test_df (trimmed for tier).
    """
    import polars as pl

    # Cache key = (tier, input-container-kind). Without the kind component the
    # cache collides between Polars and pandas inputs: in a multi-model suite
    # where Linear (non-polars-native) runs first it stashes pandas tier-DFs
    # under tier=(False,False); XGB (polars-native) later asks for the same
    # tier and gets pandas back, then XGBoostStrategy.prepare_polars_dataframe
    # raises "'DataFrame' object has no attribute 'schema'" on strategies.py:323.
    # The "kind" tag is sampled from the first non-None input, matching what
    # the caller actually passed in this invocation.
    kind = "none"
    for k in ("train_df", "val_df", "test_df"):
        v = base_dfs.get(k)
        if v is not None:
            kind = "pl" if isinstance(v, pl.DataFrame) else "pd"
            break
    tier_key = (strategy.feature_tier(), kind)
    tier = tier_key  # preserved name for downstream logging + storage
    if tier_key in tier_cache:
        return tier_cache[tier_key]

    # Determine columns to exclude for this tier
    cols_to_exclude = set()
    if text_features and not strategy.supports_text_features:
        cols_to_exclude.update(text_features)
    if embedding_features and not strategy.supports_embedding_features:
        cols_to_exclude.update(embedding_features)

    if not cols_to_exclude:
        # Tier supports all features -- use base DFs directly (no copy)
        tier_dfs = base_dfs
    else:
        if verbose:
            logger.info("  Tier %s: dropping %d text/embedding columns: %s", tier, len(cols_to_exclude), sorted(cols_to_exclude))
        tier_dfs = {}
        for key in ("train_df", "val_df", "test_df"):
            df_ = base_dfs.get(key)
            if df_ is None:
                tier_dfs[key] = None

def _split_preds_probs(arr):
    """Regression: 1-D preds; classification: 2-D probs + derived 1-D preds via argmax."""
    if arr is None:
        return None, None
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.argmax(a, axis=1), a
    return a, None

