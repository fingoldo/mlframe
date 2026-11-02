"""OOF fold fits receive the sample weight for target-transform wrappers and pipelines, as the deployed fit does."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mlframe.training._oof_weight_param import oof_weight_fit_param
from mlframe.training.trainer import _compute_oof_preds


class _WeightSpy(LinearRegression):
    """Records the weights each fit receives."""

    seen: list = []

    def fit(self, X, y, sample_weight=None):
        """Store the received weight and fit."""
        type(self).seen.append(None if sample_weight is None else np.asarray(sample_weight).copy())
        return super().fit(X, y, sample_weight=sample_weight)


class _ForwardingWrapper(TransformedTargetRegressor):
    """Declares only ``**fit_params`` like the suite's target-scaling wrapper."""

    def fit(self, X, y, **fit_params):
        """Forward everything to the base fit."""
        return super().fit(X, y, **fit_params)


def _frame():
    """Small regression frame and non-uniform weights."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    return df, df["a"].to_numpy() * 2.0, np.linspace(1.0, 5.0, 80)


def test_wrapper_and_pipeline_resolve_the_weight_keyword():
    """The keyword is found through the regressor attribute and through the final pipeline step."""
    assert oof_weight_fit_param(_ForwardingWrapper(regressor=_WeightSpy())) == "sample_weight"
    assert oof_weight_fit_param(Pipeline([("sc", StandardScaler()), ("est", _WeightSpy())])) == "est__sample_weight"
    assert oof_weight_fit_param(Pipeline([("ft", FunctionTransformer())])) is None


def test_oof_folds_of_a_wrapped_regressor_are_fit_with_weights():
    """Every OOF fold fit of a TTR-style wrapper sees a non-None weight array."""
    df, y, w = _frame()
    _WeightSpy.seen = []
    _compute_oof_preds(model=_ForwardingWrapper(regressor=_WeightSpy()), train_df=df, train_target=y, is_classifier_model=False, n_splits=3, random_seed=0, sample_weight=w)
    assert _WeightSpy.seen and all(s is not None for s in _WeightSpy.seen)
