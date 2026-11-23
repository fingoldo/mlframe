"""Cluster 41 regression: a NaN in a categorical-dtype column is mapped to the fit-time sentinel at predict (audit 2026-11-20, finding 41-F1)."""

import numpy as np
import pandas as pd

from mlframe.training.ranking.ranking import predict_ranker_scores


class _RecordingModel:
    """Stands in for a fitted CatBoost ranker and records the frame it is asked to score."""

    def __init__(self):
        self.seen = None

    def predict(self, X):
        """Record the frame and return zero scores."""
        self.seen = X.copy()
        return np.zeros(len(X))


def test_categorical_dtype_nan_is_mapped_to_sentinel_at_predict():
    """A NaN in a pandas categorical column reaches the model as the fit-time sentinel, not as NaN (41-F1)."""
    model = _RecordingModel()
    fitted = {
        "model": model,
        "flavor": "catboost",
        "cb_cat_features": ["c"],
        "cb_na_sentinel": "__MLFRAME_NA__",
    }
    X = pd.DataFrame({"c": pd.Categorical(["a", None, "b"], categories=["a", "b"]), "x": [1.0, 2.0, 3.0]})
    predict_ranker_scores(fitted, X)
    assert model.seen["c"].isna().sum() == 0
    assert "__MLFRAME_NA__" in set(model.seen["c"].astype(str))
