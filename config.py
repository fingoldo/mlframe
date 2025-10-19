from enum import Enum, auto

THOUSANDS_SEPARATOR = "_"

KERAS_MODEL_TYPES = ("Sequential",)
NGBOOST_MODEL_TYPES = ("NGBClassifier", "NGBRegressor")
LGBM_MODEL_TYPES = ("LGBMClassifier", "LGBMRegressor")
XGBOOST_MODEL_TYPES = ("XGBClassifier", "XGBRegressor")
CATBOOST_MODEL_TYPES = ("CatBoostClassifier", "CatBoostRegressor")
HGBOOST_MODEL_TYPES = ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor")
PYTORCH_MODEL_TYPES = ("PytorchLightningEstimator", "PytorchLightningRegressor", "PytorchLightningClassifier")
TABNET_MODEL_TYPES = ("TabNetClassifier", "TabNetMultiTaskClassifier", "TabNetRegressor", "TabNetMultiTaskRegressor")

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


"""            # native=
            # 1) xgboost native: dtypes are changed to categorical
            # 2) catboost native: extra params passed to the fit or model init methods (cat_features= can also be text_features/embedding_features);
            # 3) keras native:
            #     StringLookup (small or moderate # of unique values) or Hashing (lots of unique values) for string dtypes
            #        and
            #     CategoryEncoding (known # of unique values) or IntegerLookup (infinite # of unique values) for numeric dtypes
            #        (maps string to onehot, multihot,count,tfidf) + .adapt for the layer on the train set
            #     followed by Embedding (accepts only positive integers of a fixed range, that's why prev 2 are needed)"""


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
