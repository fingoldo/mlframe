"""Cluster 39 regression: failures that used to be logged at DEBUG/INFO, or gated on verbose, are visible at WARNING (audit 2026-11-20, findings 39-F8, 39-F10)."""

import logging

import numpy as np
import pandas as pd

from mlframe.training.core._volatility_lag_router import build_volatility_lag_router
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


class _Boom:
    """Model whose predict always fails."""

    def predict(self, X):
        """Raise to simulate a broken routing predict."""
        raise RuntimeError("predict broke")


def test_router_val_predict_failure_warns_with_traceback(caplog):
    """A failing val predict during routing build is WARNING, not INFO, and returns the trained model (39-F8)."""
    trained = _Boom()
    n = 120  # build_volatility_lag_router skips validation shorter than 100 rows before it predicts
    df = pd.DataFrame({"g": np.arange(n) % 4, "o": np.arange(n)})
    with caplog.at_level(logging.WARNING):
        out = build_volatility_lag_router(
            trained, _Boom(), df["g"].to_numpy(), df, np.arange(n, dtype=float), "g", "o", config=None,
        )
    assert out is trained
    assert any(r.levelno == logging.WARNING and "routing skipped" in r.getMessage() for r in caplog.records)


def test_allowed_targets_typo_warns_at_default_verbosity(caplog):
    """A misspelled allowed_targets name warns even with verbose=0 (the default) (39-F10)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x0": rng.normal(size=50), "a": rng.integers(0, 2, 50)})
    fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["a"], allowed_targets=["quaity"])
    with caplog.at_level(logging.WARNING):
        fte.build_targets(df)
    assert any("not found in built targets" in r.getMessage() for r in caplog.records)
