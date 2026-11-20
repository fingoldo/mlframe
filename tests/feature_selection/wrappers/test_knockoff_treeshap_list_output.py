"""TreeSHAP returning one array per class (older shap) must reduce over classes, not over rows."""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection.wrappers._knockoffs import knockoff_importance


def test_knockoff_treeshap_list_of_class_arrays_keeps_per_feature_axis(monkeypatch):
    """A list-of-arrays shap output yields one importance per joint column, so every |W_j| equals the unit slot gap."""
    n, p = 60, 3

    class _FakeExplainer:
        """Stand-in TreeExplainer returning a per-class list, importance of column c equal to c."""

        def __init__(self, model):
            """Accept the fitted model like shap does."""
            self.model = model

        def shap_values(self, X):
            """Return two identical class arrays whose column c holds the value c."""
            base = np.tile(np.arange(2 * p, dtype=float), (len(X), 1))
            return [base, base.copy()]

    fake = types.ModuleType("shap")
    fake.TreeExplainer = _FakeExplainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "shap", fake)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=list("abc"))
    y = (X["a"] + rng.normal(size=n) > 0).astype(int)
    W = knockoff_importance(lambda: RandomForestClassifier(n_estimators=5, random_state=0), X, y, random_state=1, w_statistic="shap")
    assert len(W) == p
    assert all(abs(abs(v) - 1.0) < 1e-12 for v in W.values())
