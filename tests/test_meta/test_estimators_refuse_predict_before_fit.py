"""The estimators named in wave-3 API3 raise ``NotFittedError`` from their predict or transform methods before fit.

They raised ``AttributeError`` on the missing fitted state, which is not the sklearn contract.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

_CASES = [
    ("mlframe.training.composite.hurdle", "HurdleRegressor", "predict"),
    ("mlframe.training.composite.quantile", "CompositeQuantileEstimator", "predict"),
    ("mlframe.training.composite.grouped_block_stacking", "GroupedBlockStacker", "predict"),
    ("mlframe.training.neural.tabular_1dcnn", "Tabular1DCNNRegressor", "predict"),
    ("mlframe.training.neural.tabular_1dcnn", "Tabular1DCNNClassifier", "predict_proba"),
    ("mlframe.training.neural.ranker", "MLPRanker", "predict"),
    ("mlframe.training.neural.trunk_residual_mlp", "TrunkResidualMLPRegressor", "predict_ensemble_mean"),
    ("mlframe.training.composite.multitask_auxiliary_loss", "MultiTaskAuxiliaryLossRegressor", "predict"),
    ("mlframe.training._classif_helpers", "_ChainEnsemble", "predict_proba"),
    ("mlframe.competition.naive_bayes_log_odds", "NaiveBayesLogOddsEnsembler", "predict_proba"),
    ("mlframe.training.composite.sklearn_compat", "CompositeTargetTransformer", "transform"),
    ("mlframe.training.composite.post_shim", "PrePipelinePredictShim", "predict"),
    ("mlframe.training.composite.per_group_router", "PerGroupCompositeRouter", "predict"),
    ("mlframe.training.neural.feature_prep", "NeuralEmbeddingTextEncoder", "transform"),
]


@pytest.mark.parametrize("module, cls_name, method", _CASES, ids=[c[1] for c in _CASES])
def test_predict_before_fit_raises_not_fitted(module, cls_name, method):
    """An unfitted instance refuses the method with NotFittedError."""
    cls = getattr(importlib.import_module(module), cls_name)
    with pytest.raises(NotFittedError):
        getattr(cls(), method)(np.zeros((5, 3)))
