"""Strict parameters of `mlframe.feature_selection.functional_adapters:GreedyBackwardEliminationSelector`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict


class GreedyBackwardEliminationParams(BaseModel):
    """Parameters of `mlframe.feature_selection.functional_adapters:GreedyBackwardEliminationSelector`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    estimator: Any = None
    scoring: Any = None
    cv: Any = None
    min_features: int = 1
    tol: float = 0.0
    n_repeats: int = 1
    seed_base: int = 0
    random_state: int = 0
