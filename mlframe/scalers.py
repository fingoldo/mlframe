# *****************************************************************************************************************************************************
# All sklearn scalers
# *****************************************************************************************************************************************************

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

ALL_SCALERS = [
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler", MinMaxScaler()),
    ("MaxAbsScaler", MaxAbsScaler()),
    (
        "RobustScaler",
        RobustScaler(quantile_range=(25, 75)),
    ),
    (
        "Yeo-Johnson scaling",
        PowerTransformer(method="yeo-johnson"),
    ),
    # ("Box-Cox scaling",        PowerTransformer(method="box-cox"),    ), # The Box-Cox transformation can only be applied to strictly positive data
    (
        "QuantileTransformerUniform",
        QuantileTransformer(
            output_distribution="uniform",
        ),
    ),
    (
        "QuantileTransformerNormal",
        QuantileTransformer(output_distribution="normal"),
    ),
    ("sample-wise L2 Normalizer", Normalizer()),
]
