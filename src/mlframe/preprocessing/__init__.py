"""Data preprocessing: cleaning, scaling, outlier handling, clustering.

Submodules:
    cleaning    - NaN / sentinel / text cleaning (was feature_cleaning.py).
    transforms  - reusable preprocessing pipeline steps.
    scalers     - scaler wrappers (RobustScaler, QuantileTransformer, ...).
    outliers    - outlier detection and capping.
    cluster     - clustering helpers used in preprocessing pipelines.
"""

from __future__ import annotations


from mlframe.preprocessing.cleaning import *
from mlframe.preprocessing.transforms import *
from mlframe.preprocessing.scalers import *
from mlframe.preprocessing.outliers import *
from mlframe.preprocessing.cluster import *
from mlframe.preprocessing.category_support import train_test_support_screen
from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift
from mlframe.preprocessing.auto_transform_select import select_column_transforms
from mlframe.preprocessing.gaussian_power_transform_search import gaussian_power_transform_search, apply_gaussian_power_transform
from mlframe.preprocessing.adversarial_rebin import adversarial_rebin_categorical
from mlframe.preprocessing.regime_conditioned_imputation import regime_conditioned_median_fill
from mlframe.preprocessing.outlier_policy import is_tree_based_model, apply_outlier_policy
from mlframe.preprocessing.degradation_augment import augment_to_match_test_distribution, match_missingness_rate, match_noise_level
from mlframe.preprocessing.rare_count_pruning import collapse_rare_categories, drop_rare_features
from mlframe.preprocessing.align_feature_direction import align_feature_direction, apply_feature_direction
from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer
from mlframe.preprocessing.sibling_group_cold_start_fill import sibling_group_cold_start_fill
from mlframe.preprocessing.missing_indicator_pairing import impute_with_missing_indicator
from mlframe.preprocessing.outlier_capping_or_missing import outlier_cap_or_missing
from mlframe.preprocessing.outlier_detector_zoo import make_outlier_detector, make_ensemble_outlier_scores, select_outlier_threshold

# Curate the star-import surface explicitly (mirrors mlframe.metrics.__init__'s pattern).
__all__ = sorted(name for name in globals() if not name.startswith("_"))
