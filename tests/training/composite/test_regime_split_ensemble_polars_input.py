"""RegimeSplitEnsemble keeps polars frames polars for every submodel (tracker 2026-11-02 wave 10, ALIGN10-5)."""

import numpy as np
import polars as pl

from mlframe.training.composite.regime_split_ensemble import RegimeSplitEnsemble


class _Recorder:
    """Regressor that records the type of the X it is fitted on and predicts a constant."""

    fit_types: list = []

    def fit(self, X, y, sample_weight=None):
        """Remember the container type."""
        _Recorder.fit_types.append(type(X).__name__)
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Constant prediction."""
        return np.full(len(X), self.m_)

    def get_params(self, deep=True):
        """No parameters, so clone() builds a fresh instance."""
        return {}

    def set_params(self, **kw):
        """No parameters."""
        return self


def test_regime_models_are_fitted_on_polars_frames():
    """Every regime model and the global model see a polars DataFrame; regime models used to get an unnamed ndarray."""
    _Recorder.fit_types = []
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(size=100), "r": rng.integers(0, 2, 100)})
    y = rng.normal(size=100)
    ens = RegimeSplitEnsemble(_Recorder, lambda X: X["r"].to_numpy(), combine="route")
    ens.fit(df, y)
    assert set(_Recorder.fit_types) == {"DataFrame"}
    assert ens.predict(df).shape == (100,)
