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
from mlframe.competition.panel_target_persistence import (
    TargetPersistenceResult,
    check_target_persistence,
    lag_target_within_group,
    lead_target_within_group,
)
from mlframe.competition.threshold_range_rescaler import (
    ThresholdRangeRescaler,
    ThresholdCorrection,
    ThresholdRangeRescalerResult,
)
from mlframe.competition.leak_scan import (
    sort_by_density_leak_scan,
    LeakScanResult,
    find_shifted_column_groups,
)
from mlframe.competition.synthetic_row_detector import (
    detect_synthetic_rows,
    count_encoding_shift_report,
    CountEncodingShiftReport,
)
from mlframe.competition.frequency_power_interaction import (
    frequency_power_interaction,
    FrequencyPowerInteractionResult,
)
from mlframe.competition.naive_bayes_log_odds import NaiveBayesLogOddsEnsembler
from mlframe.competition.known_label_override import (
    monotonic_entity_override,
    known_label_override,
)
from mlframe.competition.train_test_union_frequency import (
    train_test_union_frequency_encode,
    train_test_union_frequency_encode_hierarchical_components,
)
from mlframe.competition.logloss_clip import clip_probabilities_for_logloss
from mlframe.competition.rounded_categorical_interaction import RoundedNumericCategoricalInteraction
from mlframe.competition.trend_noise_decorrelation import inject_noise_and_recenter
from mlframe.competition.gmm_classifier import GaussianMixtureClassifier

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
    "TargetPersistenceResult",
    "check_target_persistence",
    "lag_target_within_group",
    "lead_target_within_group",
    "ThresholdRangeRescaler",
    "ThresholdCorrection",
    "ThresholdRangeRescalerResult",
    "sort_by_density_leak_scan",
    "LeakScanResult",
    "find_shifted_column_groups",
    "detect_synthetic_rows",
    "count_encoding_shift_report",
    "CountEncodingShiftReport",
    "frequency_power_interaction",
    "FrequencyPowerInteractionResult",
    "NaiveBayesLogOddsEnsembler",
    "monotonic_entity_override",
    "known_label_override",
    "train_test_union_frequency_encode",
    "train_test_union_frequency_encode_hierarchical_components",
    "clip_probabilities_for_logloss",
    "RoundedNumericCategoricalInteraction",
    "inject_noise_and_recenter",
    "GaussianMixtureClassifier",
]
