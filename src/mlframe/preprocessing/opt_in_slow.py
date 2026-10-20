"""Registry for preprocessing functions that are valid but slower than the default path.

Each entry is reached only when its name is listed in ``PreprocessingExtensionsConfig.opt_in_slow``. The default (``None``
or empty) keeps the training pipeline exactly as it was, so these functions stay off the production path unless a
caller opts in by name.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Callable, Optional

#: name -> defining module (relative to ``mlframe.preprocessing``). Each name is the function's bare name.
OPT_IN_SLOW: dict[str, str] = {
    "adversarial_rebin_categorical": "adversarial_rebin",
    "align_feature_direction": "align_feature_direction",
    "apply_feature_direction": "align_feature_direction",
    "check_feature_direction_stability": "align_feature_direction",
    "select_column_transforms": "auto_transform_select",
    "smoothed_target_encode_column": "category_support",
    "train_test_support_screen": "category_support",
    "analyse_and_clean_features": "cleaning",
    "apply_features_cleaning": "cleaning",
    "fragment_df_on_ram_usage_increase": "cleaning",
    "suggest_non_outlying_data_indices": "cleaning",
    "clusterize": "cluster",
    "augment_to_match_test_distribution": "degradation_augment",
    "apply_gaussian_power_transform": "gaussian_power_transform_search",
    "gaussian_power_transform_search": "gaussian_power_transform_search",
    "impute_with_missing_indicator": "missing_indicator_pairing",
    "outlier_cap_or_missing": "outlier_capping_or_missing",
    "make_ensemble_outlier_scores": "outlier_detector_zoo",
    "select_outlier_threshold": "outlier_detector_zoo",
    "apply_outlier_policy": "outlier_policy",
    "compute_naive_outlier_score": "outliers",
    "compute_outlier_detector_score": "outliers",
    "reject_outliers": "outliers",
    "collapse_rare_categories": "rare_count_pruning",
    "drop_rare_features": "rare_count_pruning",
    "regime_conditioned_median_fill": "regime_conditioned_imputation",
    "sibling_group_cold_start_fill": "sibling_group_cold_start_fill",
    "augment_temporal_drift": "temporal_drift_augment",
    "select_true_last_standardized": "temporal_drift_augment",
}


def resolve_opt_in_slow(name: str, enabled: Optional[Iterable[str]]) -> Callable:
    """Return the function registered under ``name`` if the caller enabled it, else raise.

    Raises ``KeyError`` for a name not in the registry and ``PermissionError`` for a registered name the config did not
    enable, so a slow path never runs by accident.
    """
    if name not in OPT_IN_SLOW:
        raise KeyError(f"{name!r} is not a registered opt-in slow preprocessing function")
    if name not in set(enabled or ()):
        raise PermissionError(f"{name!r} is opt-in slow; add it to PreprocessingExtensionsConfig.opt_in_slow to use it")
    module = importlib.import_module(f"mlframe.preprocessing.{OPT_IN_SLOW[name]}")
    fn: Callable = getattr(module, name)
    return fn
