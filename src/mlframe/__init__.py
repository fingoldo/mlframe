"""mlframe -- production-grade tabular ML toolkit (classification, regression, ranking, quantile).

Public API surface is re-exported from the subpackages below. Import the
subpackage directly (``from mlframe.metrics import ECE``) rather than reaching
into private submodules.
"""

from mlframe.version import __version__

from mlframe.config import (
    THOUSANDS_SEPARATOR,
    KERAS_MODEL_TYPES,
    LGBM_MODEL_TYPES,
    NGBOOST_MODEL_TYPES,
    XGBOOST_MODEL_TYPES,
    CATBOOST_MODEL_TYPES,
    HGBOOST_MODEL_TYPES,
    PYTORCH_MODEL_TYPES,
    TABNET_MODEL_TYPES,
    CategoricalsAssigning,
    CategoricalsHandling,
    MissingHandling,
    NumericsHandling,
    EarlyStopping,
    OutlierRemoval,
    FeatureSelection,
    HyperParameterTuning,
    SampleWeights,
    Resampling,
    TargetTransformer,
    ClassWeights,
)

__all__ = [
    "__version__",
    "THOUSANDS_SEPARATOR",
    "KERAS_MODEL_TYPES",
    "LGBM_MODEL_TYPES",
    "NGBOOST_MODEL_TYPES",
    "XGBOOST_MODEL_TYPES",
    "CATBOOST_MODEL_TYPES",
    "HGBOOST_MODEL_TYPES",
    "PYTORCH_MODEL_TYPES",
    "TABNET_MODEL_TYPES",
    "CategoricalsAssigning",
    "CategoricalsHandling",
    "MissingHandling",
    "NumericsHandling",
    "EarlyStopping",
    "OutlierRemoval",
    "FeatureSelection",
    "HyperParameterTuning",
    "SampleWeights",
    "Resampling",
    "TargetTransformer",
    "ClassWeights",
]
