"""Cluster 28 regression: a failed categorical cast on predict is a WARNING naming the column (audit 2026-11-20, finding 28-F57)."""

import logging

import pandas as pd

from mlframe.training.lgb_shim import _DatasetReuseMixin


class _Stub(_DatasetReuseMixin):
    """Model stand-in carrying only the fit-time categorical dtypes."""

    def __init__(self, train_cats):
        """Store the fit-time category dtypes."""
        self._mlframe_train_cat_dtypes = train_cats


def test_failed_predict_cast_logs_a_warning(caplog):
    """A column that cannot be cast to the train category dtype is reported at WARNING by name (28-F57)."""
    stub = _Stub({"c": pd.CategoricalDtype(["a", "b"])})
    X = pd.DataFrame({"c": [["unhashable"], ["x"]]})
    with caplog.at_level(logging.WARNING):
        stub._align_cats_for_predict(X)
    assert any(r.levelno == logging.WARNING and "'c'" in r.getMessage() for r in caplog.records)
