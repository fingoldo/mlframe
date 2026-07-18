"""Regression: multiclass SHAP importance aggregation must reduce over the
correct axes regardless of SHAP's 3-D array layout.

Modern SHAP (>=0.43) TreeExplainer returns multiclass values as
``(samples, features, classes)``; older SHAP returned
``(classes, samples, features)``. The fit-time reducer used to hard-code the
classes-first layout (``abs.sum(axis=0).mean(0)``), which on the modern layout
collapsed the importance vector to length ``n_classes`` instead of
``n_features``. That left ``Shadow_feature_import`` (the ``vals[len(X):]`` half)
empty and crashed ``update_importance_history`` with
``IndexError: index 0 is out of bounds for axis 0 with size 0`` -- surfaced by
fuzz combo ``c0095`` (hgb_xgb). The reducer now finds the feature axis by
matching ``X_boruta``'s column count, so the result is always ``n_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")


def _make_multiclass_frame(n=300, n_features=8, n_classes=3, seed=0):
    """Make multiclass frame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # A few features carry class signal so SHAP has something to attribute.
    logits = X["f0"].to_numpy() * 1.5 + X["f1"].to_numpy() * -1.0
    y = pd.Series(np.digitize(logits, np.quantile(logits, [1 / 3, 2 / 3])), name="y")
    assert y.nunique() == n_classes
    return X, y


def test_boruta_shap_multiclass_does_not_crash_on_3d_shap_layout():
    """Multiclass SHAP path (3-D shap array) must complete and yield a
    per-feature support_ of the right length, not IndexError on an empty
    Shadow_feature_import."""
    from sklearn.ensemble import RandomForestClassifier

    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _make_multiclass_frame()
    selector = BorutaShap(
        model=RandomForestClassifier(n_estimators=25, random_state=0),
        importance_measure="shap",
        classification=True,
        n_trials=6,
        sample=False,
        normalize=True,
        verbose=False,
    )
    # Pre-fix this raised IndexError mid-fit (empty Shadow_feature_import).
    selector.fit(X, y)
    assert hasattr(selector, "support_")
    # support_ being per-feature (len n_features, not n_classes) proves the
    # multiclass importance vector was reduced over the right axes: pre-fix it
    # collapsed to length n_classes (3), leaving Shadow_feature_import empty and
    # raising IndexError before support_ was ever produced.
    assert selector.support_.shape == (X.shape[1],), f"support_ must be per-feature (len {X.shape[1]}); got {selector.support_.shape}"
