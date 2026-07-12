"""COMPETITION-ONLY utilities (Kaggle-specific tricks), not for production use.

This subpackage holds exploratory/leaderboard-hunting tricks (data de-anonymization,
leak exploitation, competition-specific reverse engineering) that are useful for
Kaggle-style competitions but explicitly inapplicable or harmful in real production
ML systems (see ``MLFRAME_IDEAS_competitions.md``).

Nothing here is imported by any production mlframe module, and nothing here is
re-exported from ``mlframe``'s top-level ``__init__.py``. Import explicitly from
``mlframe.competition.<module>`` when doing competition/exploratory work.
"""

from __future__ import annotations

from mlframe.competition.value_uniqueness_encoder import value_uniqueness_encoder
from mlframe.competition.float_precision_denoise import FloatPrecisionDenoiser, DenoiseResult
from mlframe.competition.power_rescale import asymmetric_scale_by_sign, power_rescale_to_target_sum
from mlframe.competition.quantization_recovery import (
    detect_quantization_step,
    derounded_feature,
    rank_features_by_quantization_confidence,
    QuantizationRankResult,
)

__all__ = [
    *globals().get("__all__", []),
    "value_uniqueness_encoder",
    "FloatPrecisionDenoiser",
    "DenoiseResult",
    "asymmetric_scale_by_sign",
    "power_rescale_to_target_sum",
    "detect_quantization_step",
    "derounded_feature",
    "rank_features_by_quantization_confidence",
    "QuantizationRankResult",
]
