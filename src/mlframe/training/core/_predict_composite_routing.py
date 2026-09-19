"""Predict-entry-point routing for composite-target wrappers.

A ``CompositeTargetEstimator`` built by the suite reads its base column(s) from the suite-stage frame (after the shared
pipeline, row-wise and extension steps: the stage discovery fit the transform on) and applies its own inner pipeline to the
same frame for the inner model. The predict entry points must therefore hand it that stage frame untouched, never the
per-model subset / pre_pipeline output they build for raw models (the base would arrive z-scored) nor the pre-pipeline input
(the inner would miss the row-wise columns it was trained with).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


def is_composite_wrapper(model: Any) -> bool:
    """True for a composite wrapper that takes the suite-stage frame and derives the inner's input itself."""
    return bool(getattr(model, "_routes_inner_input", False))


def _required_columns(model: Any) -> list[str]:
    """Columns the wrapper reads from its frame: base columns, group column and the inner branch's fit-time inputs."""
    cols: list[str] = []
    try:
        cols.extend(model._resolve_base_columns())
    except Exception as exc:
        logger.debug("composite base-column resolution failed: %s", exc)
    if getattr(model, "group_column", None):
        cols.append(model.group_column)
    feed = getattr(model, "inner_pre_pipeline_", None) or getattr(model, "estimator_", None)
    names = getattr(feed, "feature_names_in_", None)
    if names is not None:
        cols.extend(str(c) for c in names)
    return cols


def composite_stage_frame(model: Any, df: Any, df_pre_pipeline: Any, to_pandas: Callable[[Any], Any]) -> Any:
    """Pick the frame a composite wrapper predicts on: the suite-stage ``df`` when it carries every column the wrapper reads,
    else the pre-pipeline frame when only that one does (polars-fastpath models trained on the raw frame), else ``df``.

    ``to_pandas`` converts a polars frame to the shared pandas view; the wrapper's inner pipeline is sklearn and needs pandas.
    """
    needed = _required_columns(model)
    chosen = df
    for cand in (df, df_pre_pipeline):
        if cand is None or not hasattr(cand, "columns"):
            continue
        have = {str(c) for c in cand.columns}
        if all(c in have for c in needed):
            chosen = cand
            break
    inner = getattr(model, "estimator_", None)
    if getattr(model, "inner_pre_pipeline_", None) is not None or not _is_polars_native(inner):
        chosen = to_pandas(chosen)
    return chosen


def _is_polars_native(inner: Any) -> bool:
    """Whether the inner estimator accepts a polars frame directly (CatBoost / XGBoost sklearn API)."""
    if inner is None:
        return False
    from .predict import _polars_native_class_names

    allowed = _polars_native_class_names()
    return any(cls.__name__ in allowed for cls in type(inner).__mro__)


def register_spec_transforms(metadata: Any) -> list[str]:
    """Re-register auto-chain transforms named by a loaded suite's composite specs; returns the names registered.

    Discovery registers ``chain_*`` transforms in the training process only; a fresh serving process must rebuild them before
    any wrapper or ensemble component looks them up.
    """
    if not isinstance(metadata, dict):
        return []
    names: list[str] = []
    for by_target in (metadata.get("composite_target_specs") or {}).values():
        if not isinstance(by_target, dict):
            continue
        for specs in by_target.values():
            names.extend(_spec_transform_names(specs))
    if not names:
        return []
    from ..composite.estimator._routing import ensure_transforms_registered

    return ensure_transforms_registered(names)


def _spec_transform_names(specs: Iterable[Any] | None) -> list[str]:
    """``transform_name`` of every dict spec in ``specs``."""
    return [s["transform_name"] for s in (specs or ()) if isinstance(s, dict) and isinstance(s.get("transform_name"), str)]


def original_target_of(metadata: Any, target_type: Any, target_name: Any) -> Any:
    """Original target a composite target (or its ``_CT_ENSEMBLE__`` entry) predicts; raw targets map to themselves."""
    name = str(target_name)
    if name.startswith("_CT_ENSEMBLE__"):
        return name[len("_CT_ENSEMBLE__") :]
    specs = (metadata.get("composite_target_specs") or {}) if isinstance(metadata, dict) else {}
    by_target = specs.get(str(target_type)) or specs.get(target_type) or {}
    for orig, spec_list in by_target.items() if isinstance(by_target, dict) else ():
        if any(isinstance(s, dict) and s.get("name") == name for s in spec_list or ()):
            return orig
    return target_name


def per_original_target_float_ensembles(per_target_preds: dict, metadata: Any, combine: Callable[[Any], Any]) -> dict[str, Any]:
    """Combine float predictions per ``(target_type, original target)``.

    Composite-target and CT-ensemble models predict their original target on its own y scale, so they pool with it and never
    with a different target.
    """
    import numpy as np

    groups: dict[tuple[Any, Any], list[Any]] = {}
    for (tt, tn), preds_list in per_target_preds.items():
        floats = [np.asarray(p) for p in preds_list if np.issubdtype(np.asarray(p).dtype, np.floating)]
        if floats:
            groups.setdefault((tt, original_target_of(metadata, tt, tn)), []).extend(floats)
    return {f"{tt}_{orig}": (combine(np.stack(ps)) if len(ps) > 1 else ps[0]) for (tt, orig), ps in groups.items()}
