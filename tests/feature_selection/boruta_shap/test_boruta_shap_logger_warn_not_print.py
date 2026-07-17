"""``check_missing_values`` must emit a WARNING log record, NOT stdout print -- production logs are routed via ``logging``; bare ``print`` calls inside a library bypass log filters / handlers / file sinks and pollute notebook output."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


class _DummyXGBLike:
    """Stand-in whose ``type(...).__name__`` contains the substring ``XGB``; ``check_missing_values`` matches on ``models_to_check`` via ``str(type(...))`` lowercase, so a class named with ``xgb`` in it (case-insensitive) routes to the WARNING branch instead of raising ValueError. ``feature_importances_`` attribute keeps ``check_model`` happy when ``importance_measure='gini'``."""

    feature_importances_ = np.zeros(2)

    def fit(self, X, y, **kw):
        """Helper that fit."""
        return self

    def predict(self, X):
        """Helper that predict."""
        return np.zeros(len(X))


def test_check_missing_values_logs_warning_not_print(capsys, caplog):
    """Check missing values logs warning not print."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    sel = BorutaShap(
        model=_DummyXGBLike(),
        importance_measure="gini",
        classification=True,
        n_trials=1,
        random_state=0,
        verbose=False,
    )
    # Directly populate ``X``/``y`` so we hit ``check_missing_values`` without running the full fit (which does a real shap explainer).
    sel.X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    sel.y = pd.Series([0, 1, 0])

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.boruta_shap"):
        sel.check_missing_values()

    # WARNING record present?
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "missing values" in r.getMessage().lower()]
    assert len(warning_records) >= 1, f"expected WARNING-level log on missing values; got records: {caplog.records!r}"

    # Nothing on stdout.
    captured = capsys.readouterr()
    assert "Warning" not in captured.out, f"check_missing_values must NOT print to stdout; got: {captured.out!r}"
