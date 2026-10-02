"""Strict parameters of `mlframe.feature_selection.ace:ACESelector`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict


class ACEParams(BaseModel):
    """Parameters of `mlframe.feature_selection.ace:ACESelector`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ("importance",)
    __signature_skip_default__ = ()

    estimator: Any = None
    n_replicates: int = 20
    contrast_percentile: float = 100.0
    alpha: float = 0.05
    importance: Literal["native", "permutation"] = "native"
    n_masking_rounds: int = 3
    n_perm_repeats: int = 5
    fdr_control: bool = True
    random_state: int = 0
    mask_redundant: bool = True
    masking_r2: float = 0.95
