"""ClassifierWithEarlyStopping exposes classes_ and forgets stale column names (tracker 2026-11-02 wave 14, SKL14-6)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.estimators.base import ClassifierWithEarlyStopping


def test_classes_attribute_and_stale_names():
    """cross_val_score(scoring='roc_auc') reads classes_ off the outer estimator; a refit on an ndarray drops the DataFrame names."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = (X["a"] > 0).astype(int).to_numpy()
    clf = ClassifierWithEarlyStopping(LogisticRegression())
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1]
    assert list(clf.feature_names_in_) == ["a", "b", "c"]
    clf.fit(X.to_numpy(), y)
    assert not hasattr(clf, "feature_names_in_")
