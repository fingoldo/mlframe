"""Build the deployable ``CompositeTargetEstimator`` for one trained composite-target entry.

The suite trains every composite target's inner model through the normal per-model loop, so a linear / MLP / encoded entry's
inner sees ``entry.pre_pipeline.transform(frame)`` while the spec's transform params were fit on the suite-stage frame itself.
The wrapper therefore carries the entry's fitted pipeline (applied to the inner branch only), the train-row base range for
the default-ON soft base-shrink guard, the group column of grouped transforms and the original target name. One builder is
shared by the per-model hook and the end-of-target wrap pass so both produce the same object.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def fitted_pre_pipeline(entry: Any) -> Any:
    """The entry's ``pre_pipeline`` when it is fitted, else ``None`` (an unfitted placeholder means the inner saw the raw frame)."""
    pp = getattr(entry, "pre_pipeline", None)
    if pp is None:
        return None
    try:
        from ..pipeline._pipeline_helpers import _is_fitted
    except ImportError:  # pragma: no cover - the helper ships with mlframe
        return None
    try:
        return pp if _is_fitted(pp) else None
    except Exception as exc:
        logger.warning("[CompositeTargetEstimator] could not decide whether pre_pipeline %s is fitted (%s); wrapping without it.", type(pp).__name__, exc)
        return None


def spec_base_columns(spec: dict) -> tuple[str, ...]:
    """Primary plus extra base columns of a spec, in the order the transform's params expect."""
    extra = tuple(spec.get("extra_base_columns") or ())
    return (spec["base_column"], *extra)


def train_base_values(train_df: Any, spec: dict) -> Optional[np.ndarray]:
    """Train-row base values of ``spec`` read from the suite-stage train frame, or ``None`` when a column is absent."""
    if train_df is None or not spec.get("base_column"):
        return None
    cols = spec_base_columns(spec)
    try:
        from ..composite import _extract_base_matrix

        if not all(c in train_df.columns for c in cols):
            return None
        mat = _extract_base_matrix(train_df, cols)
        return mat[:, 0] if len(cols) == 1 else mat
    except Exception as exc:
        logger.warning("[CompositeTargetEstimator] train base extraction failed for spec '%s' (%s); soft base-shrink stays off for it.", spec.get("name"), exc)
        return None


def _canonical(value: Any) -> Any:
    """Order-stable, exact (floats as hex) representation of a spec value for hashing."""
    if isinstance(value, dict):
        return [(str(k), _canonical(value[k])) for k in sorted(value, key=str)]
    if isinstance(value, np.ndarray):
        return [_canonical(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return repr(value)


def composite_spec_digest(spec: dict) -> str:
    """Digest of what defines a composite target's T: transform, base columns and fitted params.

    A cached inner model trained on a T with a different digest learned a different target, so it must be retrained rather
    than re-wrapped with the new params.
    """
    import hashlib

    payload = (str(spec.get("transform_name")), spec_base_columns(spec) if spec.get("base_column") else (), _canonical(spec.get("fitted_params") or {}))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:20]


def find_composite_spec(metadata: Any, target_type: Any, target_name: Any) -> Optional[dict]:
    """The spec whose ``name`` is ``target_name`` under ``target_type``, or ``None`` for a raw target."""
    specs = (metadata.get("composite_target_specs") or {}) if isinstance(metadata, dict) else {}
    by_target = specs.get(str(target_type)) or specs.get(target_type) or {}
    for spec_list in by_target.values() if isinstance(by_target, dict) else ():
        for spec in spec_list or ():
            if isinstance(spec, dict) and spec.get("name") == target_name:
                return spec
    return None


def build_composite_wrapper(
    *,
    entry: Any,
    inner: Any,
    spec: dict,
    y_train: np.ndarray,
    train_df: Any = None,
    target_name: Optional[str] = None,
    group_column: Optional[str] = None,
) -> Any:
    """Wrap ``inner`` (trained on the spec's T) into a y-scale predictor that takes the suite-stage frame."""
    from ..composite import CompositeTargetEstimator, get_transform

    transform = get_transform(spec["transform_name"])
    extra = tuple(spec.get("extra_base_columns") or ())
    base_columns = spec_base_columns(spec) if extra else None
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name=spec["transform_name"],
        base_column=spec["base_column"],
        base_columns=base_columns,
        transform_fitted_params=spec["fitted_params"],
        y_train=y_train,
        inner_pre_pipeline=fitted_pre_pipeline(entry),
        base_train=train_base_values(train_df, spec),
        group_column=(spec.get("group_column") or group_column) if transform.requires_groups else None,
        recurrence_continuation=bool(spec.get("recurrence_continuation", False)),
        target_name=target_name or spec.get("target_col"),
    )
    wrapper.spec_digest_ = composite_spec_digest(spec)
    return wrapper
