"""Strict parameters of `mlframe.calibration.threshold_optimizer:optimize_decision_threshold`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.calibration_params._generate`.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy
from pydantic import BaseModel, ConfigDict


class ThresholdOptimizerParams(BaseModel):
    """Parameters of `mlframe.calibration.threshold_optimizer:optimize_decision_threshold`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    n_thresholds: int = 200
    threshold_range: Tuple[float, float] = (0.0, 1.0)
    groups: Optional[numpy.ndarray] = None
    min_group_size: int = 20
    cv: Optional[int] = None
    cv_seed: int = 0
    stability_cv_threshold: float = 0.15
