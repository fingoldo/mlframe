"""Data preprocessing: cleaning, scaling, outlier handling, clustering.

Submodules:
    cleaning    - NaN / sentinel / text cleaning (was feature_cleaning.py).
    transforms  - reusable preprocessing pipeline steps.
    scalers     - scaler wrappers (RobustScaler, QuantileTransformer, ...).
    outliers    - outlier detection and capping.
    cluster     - clustering helpers used in preprocessing pipelines.
"""

from __future__ import annotations


from mlframe.preprocessing.cleaning import *  # noqa: F401,F403
from mlframe.preprocessing.transforms import *  # noqa: F401,F403
from mlframe.preprocessing.scalers import *  # noqa: F401,F403
from mlframe.preprocessing.outliers import *  # noqa: F401,F403
from mlframe.preprocessing.cluster import *  # noqa: F401,F403
