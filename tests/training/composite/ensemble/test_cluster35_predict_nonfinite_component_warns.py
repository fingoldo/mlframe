"""Audit 2026-11-20 finding 35-F17: a component that predicts NaN for some rows is named in a WARNING at predict time."""

import logging

import numpy as np

from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble


class _NaNOnSecondRow:
    """A component returning 1.0 for every row except the second, which is NaN."""

    def predict(self, X):
        """Return a constant with one NaN row."""
        out = np.full(len(X), 1.0)
        out[1] = np.nan
        return out


class _Const:
    """A component returning a constant."""

    def __init__(self, v):
        self.v = v

    def predict(self, X):
        """Return the constant for every row."""
        return np.full(len(X), self.v)


def test_nonfinite_component_prediction_is_logged(caplog):
    """The blend is NaN on the bad row, and a WARNING names the component and the count."""
    ens = CompositeCrossTargetEnsemble(
        component_models=[_NaNOnSecondRow(), _Const(2.0)],
        component_names=["bad", "good"],
        weights=np.array([0.5, 0.5]),
        strategy="weighted",
        notes={},
    )
    X = np.zeros((3, 1))
    with caplog.at_level(logging.WARNING):
        out = ens.predict(X)
    assert not np.isfinite(out[1])
    assert any("component 'bad'" in r.getMessage() and "1 non-finite" in r.getMessage() for r in caplog.records)
