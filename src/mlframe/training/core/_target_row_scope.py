"""Train one target on its labelled rows only: narrow the context to them, then put everything back.

Every ``TrainingContext`` field is classified in :data:`ROW_REGISTRY` (a meta-test fails on a field that is not):

- ``narrow:<split>``: row-aligned with one split, sliced by that split's positions in :class:`TargetRows`;
- ``index:<split>``: a split's index array, replaced by its narrowed version;
- ``full``: full-length, indexed by original row positions (targets, group ids, timestamps, weights, sequences); the
  narrowed indices already pick the right rows out of it, so it is left alone;
- ``special``: narrowed by a dedicated rule (the OD masks, the dropped high-cardinality columns);
- ``scoped_cache``: a cache that assumes the frames it saw are the frames in use, fresh for the scope;
- ``merge_back``: what training a target produces (models, metadata, logging latches); kept on exit;
- ``shared``: configuration and everything else; restored on exit like every non-merge-back field.

On exit (also on an exception) every field that is not ``merge_back`` is put back as it was, except that a frame the
body released (set to None) stays released: restoring it would pin the full frame again.

:func:`check_rows_in_scope` is the fail-closed check at the chokepoint every model fit goes through: a split index or
target slice that reaches a fit with an unlabelled row raises :class:`TargetRowScopeError`, which the continue-on-model-
failure handler re-raises instead of skipping the model.
"""

from __future__ import annotations

import contextvars
import dataclasses
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import polars as pl

from ._target_rows import TargetRows

logger = logging.getLogger(__name__)

_NARROW = {
    "train_idx": ("train_df", "train_df_polars_pre", "train_df_pd", "train_sequences"),
    "val_idx": ("val_df", "val_df_polars_pre", "val_df_pd", "val_sequences"),
    "test_idx": ("test_df", "test_df_polars_pre", "test_df_pd", "test_df_polars", "test_sequences"),
    "filtered_train_idx": ("filtered_train_df", "train_df_polars"),
    "filtered_val_idx": ("filtered_val_df", "val_df_polars"),
    "calib_idx": ("calib_df",),
}
_FULL = (
    "df", "target_by_type", "group_ids_raw", "group_ids", "timestamps", "sample_weights", "sequences", "split_row_ids", "fairness_subgroups",
)
_SPECIAL = ("train_od_idx", "val_od_idx", "outlier_detection_result", "_dropped_high_card_data", "train_df_size_bytes_cached", "val_df_size_bytes_cached")
_SCOPED_CACHES = ("_pandas_view_cache", "_recurrent_numpy_cache", "_cat_drift_implode_cache")
_MERGE_BACK = (
    "models", "ensembles", "metadata", "slug_to_original_target_type", "slug_to_original_target_name", "_all_target_audits",
    "_non_neural_train_times", "_cache_stats", "_fs_report_cache", "_model_input_fingerprint_cache", "_mrmr_identity_cache", "_pipeline_cache",
    "_pre_screen_dropped_cols", "_pre_screen_done", "_sw_log_emitted", "_val_placement_warn_emitted", "baseline_rss_mb", "artifacts",
)
# Row-aligned caches carried in ``ctx.artifacts`` across targets; a scope gets empty ones, the suite's come back on exit.
_ARTIFACT_CACHES = ("feature_side_cache", "dataset_reuse_cache")

ROW_REGISTRY: dict[str, str] = {
    **{f: f"narrow:{split}" for split, fields in _NARROW.items() for f in fields},
    **{split: f"index:{split}" for split in _NARROW},
    **{f: "full" for f in _FULL},
    **{f: "special" for f in _SPECIAL},
    **{f: "scoped_cache" for f in _SCOPED_CACHES},
    **{f: "merge_back" for f in _MERGE_BACK},
    "_row_scope": "scope",
}
_SHARED = (
    "model_name", "target_name", "preprocessing_config", "pipeline_config", "feature_types_config", "split_config", "hyperparams_config",
    "behavior_config", "reporting_config", "output_config", "outlier_detection_config", "feature_selection_config", "confidence_analysis_config",
    "baseline_diagnostics_config", "dummy_baselines_config", "quantile_regression_config", "conformal_config", "regression_calibration_config",
    "composite_target_discovery_config", "linear_model_config", "multilabel_dispatch_config", "ranking_config", "recurrent_config",
    "recurrent_models", "mlframe_models_is_default_allowlist", "verbose", "data_dir", "models_dir", "save_charts", "outlier_detector", "od_val_set",
    "use_mrmr_fs", "use_ordinary_models", "use_mlframe_ensembles", "mrmr_kwargs", "rfecv_models", "custom_pre_pipelines", "common_params_dict",
    "mlframe_models", "strategy_by_model", "sorted_mlframe_models", "additional_columns_to_drop", "df_size_mb", "train_details", "val_details",
    "test_details", "calib_details", "fairness_features", "pipeline", "extensions_pipeline", "was_polars_input", "all_models_polars_native",
    "polars_pipeline_applied", "train_df_pandas_pre_meta", "preprocessing_extensions", "cat_features", "cat_features_polars", "text_features",
    "embedding_features", "text_emb_set", "category_encoder", "imputer", "scaler", "trainset_features_stats", "defer_pandas_conv",
)
ROW_REGISTRY.update({f: "shared" for f in _SHARED})

_ACTIVE_ROWS: "contextvars.ContextVar[Optional[TargetRows]]" = contextvars.ContextVar("mlframe_active_target_rows", default=None)


class TargetRowScopeError(RuntimeError):
    """A fit was about to see a row without a label for the target in scope: a consumer bypassed the narrowing."""


def active_rows() -> Optional[TargetRows]:
    """The :class:`TargetRows` of the target being trained, or None outside a narrowed scope."""
    return _ACTIVE_ROWS.get()


def _take(value: Any, pos: np.ndarray) -> Any:
    """Rows ``pos`` of a frame, array or per-row list."""
    if isinstance(value, pd.DataFrame | pd.Series):
        return value.iloc[pos]
    if isinstance(value, pl.DataFrame | pl.Series):
        return value[pos]
    if isinstance(value, np.ndarray):
        return value[pos]
    if isinstance(value, list):
        return [value[i] for i in pos]
    raise TypeError(f"cannot narrow a {type(value).__name__} to a target's labelled rows")


def _split_pos(rows: TargetRows, split: str) -> Optional[np.ndarray]:
    """Positions to keep inside ``split``; an outlier-filtered split the suite does not have falls back to the unfiltered one."""
    if split in rows.pos:
        return rows.pos[split]
    fallback = {"filtered_train_idx": "train_idx", "filtered_val_idx": "val_idx"}.get(split)
    return rows.pos.get(fallback) if fallback else None


def _split_total(rows: TargetRows, split: str) -> Optional[int]:
    """Row count of ``split`` before narrowing, with the same fallback as :func:`_split_pos`."""
    if split in rows.n_total:
        return rows.n_total[split]
    fallback = {"filtered_train_idx": "train_idx", "filtered_val_idx": "val_idx"}.get(split)
    return rows.n_total.get(fallback) if fallback else None


def _narrowed_fields(ctx: Any, rows: TargetRows) -> dict[str, Any]:
    """New values of every narrowed field; a frame shared by two fields is sliced once."""
    out: dict[str, Any] = {}
    memo: dict[tuple[int, Any], Any] = {}
    for split, fields in _NARROW.items():
        pos = _split_pos(rows, split)
        for name in fields:
            value = getattr(ctx, name, None)
            if value is None or pos is None:
                continue
            expected = _split_total(rows, split)
            if len(value) != expected:
                raise TargetRowScopeError(f"ctx.{name} has {len(value):_} rows but {split} has {expected:_}; the registry's alignment for it is wrong")
            # Keyed by the split's index object: with outlier detection off, filtered_train_idx IS train_idx and
            # filtered_train_df IS train_df_pd, and the narrowed pair must stay one object too.
            key = (id(value), id(getattr(ctx, split, None) if getattr(ctx, split, None) is not None else split))
            if key not in memo:
                memo[key] = _take(value, pos)
            out[name] = memo[key]
        if split in rows.idx:
            out[split] = rows.idx[split]
    if "calib_idx" in rows.dropped:  # the suite's "no calib" is None, not an empty slice
        out["calib_idx"] = out["calib_df"] = None
    train_pos, val_pos = rows.pos.get("train_idx"), rows.pos.get("val_idx")
    od_masks = {"train_od_idx": train_pos, "val_od_idx": val_pos}
    for name, pos in od_masks.items():
        mask = getattr(ctx, name, None)
        if mask is not None and pos is not None:
            out[name] = np.asarray(mask)[pos]
    od_result = getattr(ctx, "outlier_detection_result", None)
    if isinstance(od_result, dict):
        out["outlier_detection_result"] = {**od_result, **{k: out[k] for k in od_masks if k in out}}
    dropped = getattr(ctx, "_dropped_high_card_data", None)
    if dropped:
        by_split = {"train": train_pos, "val": val_pos, "test": rows.pos.get("test_idx")}
        out["_dropped_high_card_data"] = {
            col: {k: (_take(v, by_split[k]) if v is not None and by_split.get(k) is not None else v) for k, v in parts.items()} for col, parts in dropped.items()
        }
    for name, split in (("train_df_size_bytes_cached", "train_idx"), ("val_df_size_bytes_cached", "val_idx")):
        size, share = getattr(ctx, name, None), rows.labelled_share(split)
        if size is not None and share is not None:
            out[name] = int(size * share)
    return out


@contextmanager
def target_row_scope(ctx: Any, rows: Optional[TargetRows]) -> Iterator[None]:
    """Narrow ``ctx`` to ``rows`` for the body, then restore it; ``rows=None`` (a fully labelled target) is a no-op."""
    if rows is None:
        yield
        return
    names = [f.name for f in dataclasses.fields(ctx)]
    saved = {name: getattr(ctx, name) for name in names}
    artifacts = getattr(ctx, "artifacts", None)
    saved_artifacts = {k: artifacts.pop(k) for k in _ARTIFACT_CACHES if isinstance(artifacts, dict) and k in artifacts}
    narrowed = _narrowed_fields(ctx, rows)
    for name, value in narrowed.items():
        setattr(ctx, name, value)
    for name in _SCOPED_CACHES:
        if isinstance(saved.get(name), dict):
            setattr(ctx, name, type(saved[name])())
    ctx._row_scope = rows
    token = _ACTIVE_ROWS.set(rows)
    try:
        yield
    finally:
        _ACTIVE_ROWS.reset(token)
        released = {name for name in narrowed if ROW_REGISTRY[name].startswith("narrow:") and getattr(ctx, name) is None}
        for name in names:
            if ROW_REGISTRY.get(name) == "merge_back":
                continue
            setattr(ctx, name, None if name in released else saved[name])
        if isinstance(artifacts, dict):
            for k in _ARTIFACT_CACHES:
                artifacts.pop(k, None)
            artifacts.update(saved_artifacts)


def _has_unlabelled(mask: np.ndarray, idx: Any) -> bool:
    """Whether split index ``idx`` holds a row outside ``mask``."""
    arr = np.asarray(idx)
    if arr.size == 0:
        return False
    return not bool(mask[arr].all()) if arr.dtype.kind in "iu" else not bool(mask[np.flatnonzero(arr)].all())


def check_rows_in_scope(common_params: Optional[dict]) -> None:
    """Refuse a fit whose split indices or target slices include a row without a label for the target in scope."""
    rows = _ACTIVE_ROWS.get()
    if rows is None or not common_params:
        return
    for key in ("train_idx", "val_idx", "test_idx", "calib_idx"):
        idx = common_params.get(key)
        if idx is not None and _has_unlabelled(rows.mask, idx):
            raise TargetRowScopeError(f"{key} reaches a model fit with rows that have no label for this target (rows {rows.signature})")
    for key in ("train_target", "val_target", "test_target", "calib_target"):
        y = common_params.get(key)
        if y is None:
            continue
        values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        if values.dtype.kind in "fO" and pd.isna(values).any():
            raise TargetRowScopeError(f"{key} reaches a model fit with missing labels (rows {rows.signature})")
