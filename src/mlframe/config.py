"""Shared constants and enums describing the ML pipeline configuration surface: supported model type-name tuples per backend, and the enums controlling how categoricals/missing values/numerics/early stopping/feature selection/HPT/sample weights/resampling/target transforms/class weights are handled natively vs via a wrapper."""

from __future__ import annotations

from enum import Enum, auto

THOUSANDS_SEPARATOR = "_"

KERAS_MODEL_TYPES = ("Sequential",)
LGBM_MODEL_TYPES = ("LGBMClassifier", "LGBMRegressor")
NGBOOST_MODEL_TYPES = ("NGBClassifier", "NGBRegressor")
XGBOOST_MODEL_TYPES = (
    "XGBClassifier",
    "XGBRegressor",
    # 2026-04-24 DMatrix-reuse shims (mlframe.xgb_dmatrix_reuse_shim).
    # Subclass XGBClassifier / XGBRegressor; without listing them here
    # every ``model_type_name in XGBOOST_MODEL_TYPES`` check downstream
    # silently skips XGB-specific handling for the shim variants
    # (fuzz c0102 / 2026-04-27 - the XGB cat-with-float-dtype recast guard
    # never fired because the type-name check rejected the shim).
    "XGBClassifierWithDMatrixReuse",
    "XGBRegressorWithDMatrixReuse",
)
CATBOOST_MODEL_TYPES = ("CatBoostClassifier", "CatBoostRegressor")
HGBOOST_MODEL_TYPES = ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor")
PYTORCH_MODEL_TYPES = ("PytorchLightningEstimator", "PytorchLightningRegressor", "PytorchLightningClassifier")
TABNET_MODEL_TYPES = ("TabNetClassifier", "TabNetMultiTaskClassifier", "TabNetRegressor", "TabNetMultiTaskRegressor")

# ----------------------------------------------------------------------------------------------------------------------------
# ML pipelines block enums
# ----------------------------------------------------------------------------------------------------------------------------


class CategoricalsAssigning(Enum):
    """Controls which columns get flagged as categorical before backend-specific handling is decided."""

    NativeOnly = auto()  # only explicitly denoted as categorical
    NativeAndPotential = auto()  # plus potentially categorical, as found by analyse_and_clean_features in potentially_categorical_features
    All = auto()  # All features converted to Category. Use KBinsDiscretizer where needed.


class CategoricalsHandling(Enum):
    """Strategy for turning categorical columns into a form each backend can consume, falling back through native support, category-encoding (CE), or dropping."""

    Drop = auto()
    NativeOrDrop = auto()  # drop cat columns if no native possible
    NativeOrCE = auto()  # CE cat columns if no native possible
    CE = auto()


# native= handling per backend:
# 1) xgboost native: dtypes are changed to categorical
# 2) catboost native: extra params passed to the fit or model init methods (cat_features= can also be text_features/embedding_features);
# 3) keras native:
#     StringLookup (small or moderate # of unique values) or Hashing (lots of unique values) for string dtypes
#        and
#     CategoryEncoding (known # of unique values) or IntegerLookup (infinite # of unique values) for numeric dtypes
#        (maps string to onehot, multihot,count,tfidf) + .adapt for the layer on the train set
#     followed by Embedding (accepts only positive integers of a fixed range, that's why prev 2 are needed)


class MissingHandling(Enum):
    """Strategy for columns with missing values: drop them, rely on backend-native NaN handling, fall back to sklearn imputation, or always impute via sklearn."""

    Drop = auto()
    NativeOrDrop = auto()  # drop columns with missing values if no native possible
    NativeOrSklearn = auto()  # impute columns with missing values if no native possible
    Sklearn = auto()


class NumericsHandling(Enum):
    """Strategy for numeric columns: drop them, pass through unchanged, or route through sklearn FE/scaling/normalization/discretization."""

    Drop = auto()
    AsIs = auto()
    Sklearn = auto()  # FE;Scale;Normalize;Discretize in any combination


class EarlyStopping(Enum):
    """Whether/how early stopping is applied: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class OutlierRemoval(Enum):  # OR
    """Whether/how outlier removal is applied before training: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class FeatureSelection(Enum):  # FS
    """Whether/how feature selection is applied: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class HyperParameterTuning(Enum):  # HPT
    """Whether/how hyperparameter tuning is applied: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class SampleWeights(Enum):  # SW
    """Whether/how per-sample weights are supplied to the estimator: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class Resampling(Enum):  # RS
    """Whether/how imbalance resampling (over/under-sampling) is applied: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class TargetTransformer(Enum):  # TT
    """Whether/how a target transform (e.g. log/quantile) is applied before fitting: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class ClassWeights(Enum):  # CW
    """Whether/how class weighting for imbalanced targets is applied: disabled, backend-native-or-disabled, or backend-native-or-external-wrapper."""

    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()
