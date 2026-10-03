"""Strict sub-configs of the calibration steps: ``threshold_optimizer_kwargs``, ``isotonic_risk_kwargs`` and ``confidence_shrinkage_kwargs``.

Each extends the model generated from the callee's signature, so an unknown name, a wrong type or an out-of-range value raises when the config
is created instead of deep inside finalize (where a failure is logged and the step skipped). ``to_kwargs`` returns only the fields the caller
wrote, so everything else keeps the callee's own default. Dicts are still accepted wherever these are used.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import Field

from .._sparse_params import SparseParamsModel
from .confidence_shrinkage import ConfidenceShrinkageParams
from .isotonic_risk import IsotonicRiskParams
from .threshold_optimizer import ThresholdOptimizerParams


class ThresholdOptimizerConfig(SparseParamsModel, ThresholdOptimizerParams):
    """``optimize_decision_threshold`` parameters, plus ``metric_fn`` (scores ``(y_true, y_pred)``; the finalize phase uses balanced accuracy when it is not set)."""

    metric_fn: Optional[Callable[[Any, Any], float]] = None
    n_thresholds: int = Field(default=200, ge=2)
    min_group_size: int = Field(default=20, ge=1)
    cv: Optional[int] = Field(default=None, ge=2)


class IsotonicRiskConfig(SparseParamsModel, IsotonicRiskParams):
    """``isotonic_overfit_risk`` parameters."""

    segment_ratio_threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    density_window: float = Field(default=0.05, gt=0.0, le=1.0)


class ConfidenceShrinkageConfig(SparseParamsModel, ConfidenceShrinkageParams):
    """``apply_confidence_shrinkage`` parameters, plus ``segment_ids`` for ``compute_oof_confidence`` (per-row segment labels of the OOF predictions)."""

    segment_ids: Optional[Any] = None
    min_confidence: float = Field(default=1.0, ge=0.0)
