from enum import Enum, auto


THOUSANDS_SEPARATOR = "_"


KERAS_MODEL_TYPES = ("Sequential",)
CATBOOST_MODEL_TYPES = ("CatBoostClassifier", "CatBoostRegressor")
XGBOOST_MODEL_TYPES = ("XGBClassifier", "XGBRegressor")


# ----------------------------------------------------------------------------------------------------------------------------
# ML pipelines block enums
# ----------------------------------------------------------------------------------------------------------------------------


class CategoricalsAssigning(Enum):
    NativeOnly = auto()  # only explicitly denoted as categorical
    NativeAndPotential = auto()  # plus potentially categorical, as found by analyse_and_clean_features in potentially_categorical_features
    All = auto()  # All features converted to Category. Use KBinsDiscretizer where needed.


class CategoricalsHandling(Enum):
    Drop = auto()
    NativeOrDrop = auto()  # drop cat columns if no native possible
    NativeOrCE = auto()  # CE cat columns if no native possible
    CE = auto()
    # NativeOrSklearn = auto()  # CE cat columns if no native possible
    # Sklearn = auto()


class MissingHandling(Enum):
    Drop = auto()
    NativeOrDrop = auto()  # drop columns with missing values if no native possible
    NativeOrSklearn = auto()  # impute columns with missing values if no native possible
    Sklearn = auto()


class NumericsHandling(Enum):
    Drop = auto()
    AsIs = auto()
    Sklearn = auto()  # FE;Scale;Normalize;Discretize in any combination


class EarlyStopping(Enum):
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class OutlierRemoval(Enum):  # OR
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class FeatureSelection(Enum):  # FS
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class HyperParameterTuning(Enum):  # HPT
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class SampleWeights(Enum):  # SW
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class Resampling(Enum):  # RS
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class TargetTransformer(Enum):  # TT
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()


class ClassWeights(Enum):  # CW
    No = auto()
    NativeOrNo = auto()
    NativeOrWrapper = auto()
    