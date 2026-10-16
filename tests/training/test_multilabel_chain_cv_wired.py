"""MultilabelDispatchConfig.cv must reach every ClassifierChain the chain ensemble fits."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.training._classif_helpers import _maybe_wrap_multilabel
from mlframe.training.configs import MultilabelDispatchConfig, TargetTypes


def _data(n=120, k=3, seed=0):
    """Small correlated multilabel problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = np.column_stack([(X[:, j % 4] + 0.3 * rng.normal(size=n) > 0).astype(int) for j in range(k)])
    return X, y


@pytest.mark.parametrize("cv", [2, 3, None])
def test_cv_reaches_the_fitted_chains(cv):
    """The ensemble and each fitted ClassifierChain carry the configured cv, not a hardcoded 5."""
    X, y = _data()
    cfg = MultilabelDispatchConfig(strategy="chain", n_chains=2, cv=cv)
    model = _maybe_wrap_multilabel(LogisticRegression(max_iter=200), TargetTypes.MULTILABEL_CLASSIFICATION, cfg, False, n_labels=y.shape[1])
    assert model.cv == cv
    model.fit(X, y)
    assert [c.cv for c in model.chains_] == [cv, cv]


def test_non_default_cv_no_longer_warns_as_inert():
    """Negative control for the old INERT_FIELDS entry: a non-default cv is consumed, so constructing the config is silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MultilabelDispatchConfig(strategy="chain", cv=3)
