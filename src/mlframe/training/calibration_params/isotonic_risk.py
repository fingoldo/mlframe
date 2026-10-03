"""Strict parameters of `mlframe.calibration.isotonic_risk:isotonic_overfit_risk`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.calibration_params._generate`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class IsotonicRiskParams(BaseModel):
    """Parameters of `mlframe.calibration.isotonic_risk:isotonic_overfit_risk`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    segment_ratio_threshold: float = 0.05
    remediate: bool = False
    density_window: float = 0.05
