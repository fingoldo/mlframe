"""importance_measure='gini' must accept an unfitted model whose feature_importances_ exists once fitted."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlframe.feature_selection.boruta_shap import BorutaShap


def _xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=150) for i in range(4)})
    return X, (X["f0"] > 0).astype(int)


@pytest.mark.parametrize("make", [
    lambda: DecisionTreeClassifier(random_state=0),
    lambda: pytest.importorskip("lightgbm").LGBMClassifier(n_estimators=10, verbose=-1),
])
def test_tree_models_run_with_gini(make):
    """Before fit the property raises NotFittedError, so hasattr() said "no importances" and every such model was refused."""
    X, y = _xy()
    selector = BorutaShap(model=make(), importance_measure="gini", classification=True, n_trials=2, random_state=0).fit(X, y)
    assert hasattr(selector, "accepted"), "the fit must complete and publish its decisions"


def test_a_model_without_importances_is_still_refused():
    X, y = _xy()
    with pytest.raises(AttributeError, match="feature_importances_"):
        BorutaShap(model=LogisticRegression(), importance_measure="gini", classification=True, n_trials=2).fit(X, y)
