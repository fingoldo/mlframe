"""Strict parameters of `mlframe.calibration.confidence_shrinkage:apply_confidence_shrinkage`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.calibration_params._generate`.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy
from pydantic import BaseModel, ConfigDict


class ConfidenceShrinkageParams(BaseModel):
    """Parameters of `mlframe.calibration.confidence_shrinkage:apply_confidence_shrinkage`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    neutral_value: float = 0.5
    min_confidence: float = 1.0
    max_confidence: Optional[float] = None
    segments: Optional[Dict[str, numpy.ndarray]] = None
