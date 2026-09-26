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


def composite_predict(model: Any, model_obj: Any, df: Any, df_pre_pipeline: Any, to_pandas: Callable[[Any], Any]) -> Any:
    """Predict with a composite wrapper from the suite-stage frame.

    A wrapper that does not carry its inner's pipeline (built before the suite stamped it, or by a caller of
    ``from_fitted_inner``) still needs the inner fed through the entry's fitted ``pre_pipeline``, so that stage is computed here
    and passed as ``inner_X`` while the base stays on the raw frame.
    """
    import numpy as np

    stage = composite_stage_frame(model, df, df_pre_pipeline, to_pandas)
    inner_X = None
    if getattr(model, "inner_pre_pipeline_", None) is None:
        pp = getattr(model_obj, "pre_pipeline", None)
        if pp is not None and _is_fitted_pipeline(pp):
            from ..composite.post_shim import subset_to_fit_columns

            inner_X = subset_to_fit_columns(pp.transform(subset_to_fit_columns(stage, pp)), getattr(model, "estimator_", None))
    return np.asarray(model.predict(stage) if inner_X is None else model.predict(stage, inner_X=inner_X))


def _is_fitted_pipeline(pp: Any) -> bool:
    """Whether ``pp`` is a fitted transformer (an unfitted placeholder means the inner was trained on the frame as it stands)."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(pp)
        return True
    except NotFittedError as exc:
        logger.debug("check_is_fitted(pre_pipeline) says unfitted: %s", exc)
        return False
    except TypeError as exc:
        # Not an sklearn estimator at all, so fittedness cannot be read. Treating it as unfitted routes the frame past
        # it untransformed, which is only right if it really is a placeholder -- say so rather than decide silently.
        logger.warning("pre_pipeline %s is not an sklearn estimator (%s); treating it as an unfitted placeholder", type(pp).__name__, exc)
        return False


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
    from mlframe.training.composite.estimator.shared import ensure_transforms_registered

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
