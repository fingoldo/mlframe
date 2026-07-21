"""COMPETITION-ONLY utilities (Kaggle-specific tricks), not for production use.

This subpackage holds exploratory/leaderboard-hunting tricks (data de-anonymization,
leak exploitation, competition-specific reverse engineering) that are useful for
Kaggle-style competitions but explicitly inapplicable or harmful in real production
ML systems (see ``MLFRAME_IDEAS_competitions.md``).

Nothing here is imported by any production mlframe module, and nothing here is
re-exported from ``mlframe``'s top-level ``__init__.py``. Import explicitly from
``mlframe.competition.<module>`` when doing competition/exploratory work.

Every public entry point below (a function or class actually invoked to run a
trick, as opposed to a plain result/dataclass container) is patched in place so
its FIRST call/instantiation in a process emits a one-time ``UserWarning`` --
this catches accidental production wiring at runtime, not just at code review,
without every one of the 16 trick modules needing to re-implement its own
warning banner. The patch is applied directly to the callable's home module
attribute (not just the name re-exported here), so it fires regardless of
whether a caller imports via ``mlframe.competition.X`` or
``mlframe.competition.<module>.X``.
"""

from __future__ import annotations

import functools
import warnings

_WARNED_COMPETITION_ONLY: set[str] = set()


def _warn_competition_only_once(qualname: str) -> None:
    """Emits the COMPETITION/EXPLORATORY-ONLY UserWarning for ``qualname``, once per process."""
    if qualname not in _WARNED_COMPETITION_ONLY:
        _WARNED_COMPETITION_ONLY.add(qualname)
        warnings.warn(
            f"mlframe.competition.{qualname} is a COMPETITION/EXPLORATORY-ONLY utility (Kaggle-specific "
            "trick), never intended for production pipelines -- see the mlframe.competition package docstring.",
            UserWarning,
            stacklevel=3,
        )


def _mark_competition_only_function(module: object, func_name: str) -> None:
    """Patch ``module.<func_name>`` in place so its first call in this process emits a one-time warning."""
    original = getattr(module, func_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        """Emits the one-time competition-only warning, then delegates to the original function."""
        _warn_competition_only_once(func_name)
        return original(*args, **kwargs)

    setattr(module, func_name, wrapped)


def _mark_competition_only_class(cls: type) -> type:
    """Patch ``cls.__init__`` in place so the class's first instantiation in this process emits a one-time warning."""
    original_init = cls.__init__  # type: ignore[misc]  # intentional __init__ monkeypatch for the one-time-warning wrapper

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        """Emits the one-time competition-only warning, then delegates to the original __init__."""
        _warn_competition_only_once(cls.__name__)
        return original_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init  # type: ignore[misc]  # same intentional monkeypatch as above
    return cls


from mlframe.competition import value_uniqueness_encoder as _value_uniqueness_encoder_mod
from mlframe.competition import float_precision_denoise as _float_precision_denoise_mod
from mlframe.competition import power_rescale as _power_rescale_mod
from mlframe.competition import quantization_recovery as _quantization_recovery_mod
from mlframe.competition import panel_target_persistence as _panel_target_persistence_mod
from mlframe.competition import threshold_range_rescaler as _threshold_range_rescaler_mod
from mlframe.competition import leak_scan as _leak_scan_mod
from mlframe.competition import synthetic_row_detector as _synthetic_row_detector_mod
from mlframe.competition import frequency_power_interaction as _frequency_power_interaction_mod
from mlframe.competition import naive_bayes_log_odds as _naive_bayes_log_odds_mod
from mlframe.competition import known_label_override as _known_label_override_mod
from mlframe.competition import train_test_union_frequency as _train_test_union_frequency_mod
from mlframe.competition import logloss_clip as _logloss_clip_mod
from mlframe.competition import rounded_categorical_interaction as _rounded_categorical_interaction_mod
from mlframe.competition import trend_noise_decorrelation as _trend_noise_decorrelation_mod
from mlframe.competition import gmm_classifier as _gmm_classifier_mod

for _mod, _func_names in (
    (_value_uniqueness_encoder_mod, ["value_uniqueness_encoder"]),
    (_power_rescale_mod, ["asymmetric_scale_by_sign", "power_rescale_to_target_sum"]),
    (_quantization_recovery_mod, ["detect_quantization_step", "derounded_feature", "rank_features_by_quantization_confidence"]),
    (_panel_target_persistence_mod, ["check_target_persistence", "lag_target_within_group", "lead_target_within_group"]),
    (_leak_scan_mod, ["sort_by_density_leak_scan", "find_shifted_column_groups"]),
    (_synthetic_row_detector_mod, ["detect_synthetic_rows", "count_encoding_shift_report"]),
    (_frequency_power_interaction_mod, ["frequency_power_interaction"]),
    (_known_label_override_mod, ["monotonic_entity_override", "known_label_override"]),
    (_train_test_union_frequency_mod, ["train_test_union_frequency_encode", "train_test_union_frequency_encode_hierarchical_components"]),
    (_logloss_clip_mod, ["clip_probabilities_for_logloss"]),
    (_trend_noise_decorrelation_mod, ["inject_noise_and_recenter"]),
):
    for _func_name in _func_names:
        _mark_competition_only_function(_mod, _func_name)

for _cls_mod, _cls_name in (
    (_float_precision_denoise_mod, "FloatPrecisionDenoiser"),
    (_threshold_range_rescaler_mod, "ThresholdRangeRescaler"),
    (_naive_bayes_log_odds_mod, "NaiveBayesLogOddsEnsembler"),
    (_rounded_categorical_interaction_mod, "RoundedNumericCategoricalInteraction"),
    (_gmm_classifier_mod, "GaussianMixtureClassifier"),
):
    _mark_competition_only_class(getattr(_cls_mod, _cls_name))

value_uniqueness_encoder = _value_uniqueness_encoder_mod.value_uniqueness_encoder
FloatPrecisionDenoiser = _float_precision_denoise_mod.FloatPrecisionDenoiser
DenoiseResult = _float_precision_denoise_mod.DenoiseResult
asymmetric_scale_by_sign = _power_rescale_mod.asymmetric_scale_by_sign
power_rescale_to_target_sum = _power_rescale_mod.power_rescale_to_target_sum
detect_quantization_step = _quantization_recovery_mod.detect_quantization_step
derounded_feature = _quantization_recovery_mod.derounded_feature
rank_features_by_quantization_confidence = _quantization_recovery_mod.rank_features_by_quantization_confidence
QuantizationRankResult = _quantization_recovery_mod.QuantizationRankResult
TargetPersistenceResult = _panel_target_persistence_mod.TargetPersistenceResult
check_target_persistence = _panel_target_persistence_mod.check_target_persistence
lag_target_within_group = _panel_target_persistence_mod.lag_target_within_group
lead_target_within_group = _panel_target_persistence_mod.lead_target_within_group
ThresholdRangeRescaler = _threshold_range_rescaler_mod.ThresholdRangeRescaler
ThresholdCorrection = _threshold_range_rescaler_mod.ThresholdCorrection
ThresholdRangeRescalerResult = _threshold_range_rescaler_mod.ThresholdRangeRescalerResult
sort_by_density_leak_scan = _leak_scan_mod.sort_by_density_leak_scan
LeakScanResult = _leak_scan_mod.LeakScanResult
find_shifted_column_groups = _leak_scan_mod.find_shifted_column_groups
detect_synthetic_rows = _synthetic_row_detector_mod.detect_synthetic_rows
count_encoding_shift_report = _synthetic_row_detector_mod.count_encoding_shift_report
CountEncodingShiftReport = _synthetic_row_detector_mod.CountEncodingShiftReport
frequency_power_interaction = _frequency_power_interaction_mod.frequency_power_interaction
FrequencyPowerInteractionResult = _frequency_power_interaction_mod.FrequencyPowerInteractionResult
NaiveBayesLogOddsEnsembler = _naive_bayes_log_odds_mod.NaiveBayesLogOddsEnsembler
monotonic_entity_override = _known_label_override_mod.monotonic_entity_override
known_label_override = _known_label_override_mod.known_label_override
train_test_union_frequency_encode = _train_test_union_frequency_mod.train_test_union_frequency_encode
train_test_union_frequency_encode_hierarchical_components = _train_test_union_frequency_mod.train_test_union_frequency_encode_hierarchical_components
clip_probabilities_for_logloss = _logloss_clip_mod.clip_probabilities_for_logloss
RoundedNumericCategoricalInteraction = _rounded_categorical_interaction_mod.RoundedNumericCategoricalInteraction
inject_noise_and_recenter = _trend_noise_decorrelation_mod.inject_noise_and_recenter
GaussianMixtureClassifier = _gmm_classifier_mod.GaussianMixtureClassifier

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
