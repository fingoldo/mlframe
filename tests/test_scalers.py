"""Round-trip + downstream-AUROC smoke tests for the scaler list the user
specified in the Audit #02 plan. Kept dependency-light (no torch chain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

try:
    from tests.conftest import fast_subset
except ImportError:  # pragma: no cover
    def fast_subset(values, **_):
        return list(values)


ALL_SCALERS = [
    pytest.param(RobustScaler(), id="RobustScaler"),
    pytest.param(StandardScaler(with_mean=False), id="StandardScaler_nomean"),
    pytest.param(StandardScaler(with_mean=True), id="StandardScaler"),
    pytest.param(PowerTransformer(method="yeo-johnson", standardize=True), id="PowerTransformer_yj"),
    pytest.param(PowerTransformer(method="yeo-johnson", standardize=False), id="PowerTransformer_yj_nostd"),
    pytest.param(QuantileTransformer(output_distribution="uniform", n_quantiles=100), id="QuantileTransformer_uniform"),
    pytest.param(QuantileTransformer(output_distribution="normal", n_quantiles=100), id="QuantileTransformer_normal"),
    pytest.param(MinMaxScaler(feature_range=(1, 2)), id="MinMaxScaler_12"),
]


@pytest.mark.parametrize("scaler", fast_subset(ALL_SCALERS, representative="StandardScaler"))
def test_scaler_round_trip_and_auroc(scaler):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    assert X_tr_s.shape == X_tr.shape
    assert np.isfinite(X_tr_s).all() and np.isfinite(X_te_s).all()
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, proba)
    assert auc >= 0.95, f"auroc too low for {scaler}: {auc:.3f}"
