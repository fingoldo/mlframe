"""Strict parameters of `mlframe.feature_selection.functional_adapters:ForwardSelectSelector`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class ForwardSelectParams(BaseModel):
    """Parameters of `mlframe.feature_selection.functional_adapters:ForwardSelectSelector`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    estimator: Any = None
    scoring: Any = None
    cv: int = 5
    max_features: Optional[int] = None
    min_improvement: float = 0.0
    patience: Optional[int] = None
    significance_level: float = 0.05
    random_state: int = 0
