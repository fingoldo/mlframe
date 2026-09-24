"""In train_or_test='test' mode SHAP must explain the 70% slice the model was fitted on, not the full frame."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
lightgbm = pytest.importorskip("lightgbm")

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_the_explained_rows_are_the_training_slice(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=200) for i in range(4)})
    y = (X["f0"] > 0).astype(int)
    seen = {}

    class _Spy:
        def __init__(self, model, *a, **k):
            self.model = model

        def shap_values(self, data, *a, **k):
            seen["rows"] = len(data)
            return np.zeros((len(data), data.shape[1]))

    import shap

    monkeypatch.setattr(shap, "TreeExplainer", _Spy)
    selector = BorutaShap(
        model=lightgbm.LGBMClassifier(n_estimators=10, verbose=-1), importance_measure="shap", classification=True,
        n_trials=1, train_or_test="test", random_state=0,
    )
    selector.fit(X, y)
    assert seen, "the explainer was never called"
    assert seen["rows"] == len(selector.X_boruta_train), "the explanation must run on the rows the model was fitted on"
