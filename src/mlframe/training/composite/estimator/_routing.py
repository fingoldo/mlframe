"""Input routing and transform resolution for ``CompositeTargetEstimator``.

A composite wrapper needs two different views of one row: the base column(s) at the stage the transform params were fit on
(the suite-stage frame discovery saw), and the feature frame the inner model was trained on, which for linear / MLP / encoded
strategies is that same frame pushed through the entry's own fitted ``pre_pipeline``. ``inner_input`` derives the second view
from the first, so every caller hands the wrapper ONE frame (the suite-stage one) and gets a correct y-scale prediction.

``resolve_transform`` also re-creates auto-discovered ``chain_*`` transforms that exist only in the registry of the process that
ran discovery, so a wrapper unpickled in a fresh interpreter can still find its transform.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from ..post_shim import subset_to_fit_columns
from ..transforms import Transform, UnknownTransformError, get_transform

logger = logging.getLogger(__name__)


def ensure_transforms_registered(transform_names: Iterable[str] | None) -> list[str]:
    """Register every auto-chain name in ``transform_names`` that the live registry lacks; return the names registered.

    Auto-chain transforms are added to the registry at discovery time only, so a model loaded in another process refers to a
    name nothing has registered. Non-chain names and names already present are left alone.
    """
    names = [n for n in (transform_names or ()) if isinstance(n, str) and n.startswith("chain_")]
    if not names:
        return []
    from ..discovery._auto_chain import reregister_auto_chain_transforms

    return reregister_auto_chain_transforms(names)


def resolve_transform(name: str) -> Transform:
    """``get_transform`` that rebuilds a missing auto-chain transform instead of raising ``UnknownTransformError``."""
    try:
        return get_transform(name)
    except UnknownTransformError:
        if not ensure_transforms_registered([name]):
            raise
        return get_transform(name)


def inner_input(self: Any, X: Any, transform: Transform) -> Any:
    """Build the frame the inner estimator was trained on from the suite-stage frame ``X``.

    Drops the group plumbing column of grouped transforms, applies the entry's fitted ``inner_pre_pipeline_`` (set by
    ``from_fitted_inner`` when the suite trained the inner behind a scaler / imputer / encoder), then subsets to the inner's own
    fit-time columns so a wider frame (extra suite columns, base columns the inner never saw) is accepted.
    """
    X_in = X
    drop_group = bool(transform.requires_groups and self.group_column)
    if drop_group:
        X_in = self._drop_columns(X_in, [self.group_column])
    pp = getattr(self, "inner_pre_pipeline_", None)
    if pp is not None:
        from ...core._prediction_memo import memo_transform

        _raw = X_in
        # Inside a composite post-processing phase the same frame reaches the same fitted pipeline from several callers.
        X_in = memo_transform(pp, X, lambda: pp.transform(subset_to_fit_columns(_raw, pp)), tag="drop_group" if drop_group else "")
    return subset_to_fit_columns(X_in, self.estimator_)
