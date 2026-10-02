"""Strict parameters of `mlframe.feature_selection.functional_adapters:CascadeSelectSelector`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class CascadeSelectParams(BaseModel):
    """Parameters of `mlframe.feature_selection.functional_adapters:CascadeSelectSelector`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    estimator: Any = None
    n_boruta_iterations: int = 20
    boruta_alpha: float = 0.05
    forward_max_features: Optional[int] = None
    forward_min_improvement: float = 0.0
    cv: int = 5
    scoring: Any = None
    random_state: int = 42
    rfecv_kwargs: Optional[dict] = None
