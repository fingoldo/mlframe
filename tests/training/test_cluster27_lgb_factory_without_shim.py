"""Audit 2026-11-20 finding 27-F15: with the dataset-reuse shim off, the LightGBM factories return the plain LightGBM classes.

They used to return the module-level name, which is never bound, so the factory handed back None. The test forces the shim off and
checks the factories return the lightgbm classes.
"""

import pytest

pytest.importorskip("lightgbm")

import lightgbm

from mlframe.training import _model_factories as mf


def test_factories_return_lightgbm_classes_when_the_shim_is_off(monkeypatch):
    """With USE_LGB_DATASET_REUSE_SHIM False both factories return the lightgbm classes, not None."""
    monkeypatch.setattr(mf, "USE_LGB_DATASET_REUSE_SHIM", False)
    assert mf._lgb_classifier_cls() is lightgbm.LGBMClassifier
    assert mf._lgb_regressor_cls() is lightgbm.LGBMRegressor
