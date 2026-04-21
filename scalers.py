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

# Factories — each entry returns a *fresh* stateful scaler on call.
# Prior revision exposed a module-level list of pre-instantiated scalers; any
# `.fit()` then polluted state visible to every other caller. Use a factory
# pair `(name, callable)` so consumers always get a clean instance.
_SCALER_FACTORIES = [
    ("StandardScaler", StandardScaler),
    ("MinMaxScaler", MinMaxScaler),
    ("MaxAbsScaler", MaxAbsScaler),
    ("RobustScaler", lambda: RobustScaler(quantile_range=(25, 75))),
    ("Yeo-Johnson scaling", lambda: PowerTransformer(method="yeo-johnson")),
    # Box-Cox scaling excluded: only applies to strictly positive data.
    ("QuantileTransformerUniform", lambda: QuantileTransformer(output_distribution="uniform")),
    ("QuantileTransformerNormal", lambda: QuantileTransformer(output_distribution="normal")),
    ("sample-wise L2 Normalizer", Normalizer),
]


def make_all_scalers():
    """Return a fresh list of ``(name, scaler_instance)`` tuples.

    Each call builds brand-new unfitted scalers so callers never share state.
    """
    return [(name, factory()) for name, factory in _SCALER_FACTORIES]


def __getattr__(name):
    # Backward-compat: `from mlframe.scalers import ALL_SCALERS` now yields
    # fresh instances every access rather than a shared module-level list.
    if name == "ALL_SCALERS":
        return make_all_scalers()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["make_all_scalers", "ALL_SCALERS"]
