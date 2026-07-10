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
from mlframe.preprocessing.adversarial_rebin import adversarial_rebin_categorical
from mlframe.preprocessing.regime_conditioned_imputation import regime_conditioned_median_fill
from mlframe.preprocessing.outlier_policy import is_tree_based_model, apply_outlier_policy
