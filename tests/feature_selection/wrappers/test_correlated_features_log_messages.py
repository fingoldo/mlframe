"""CorrelatedFeaturesSelector's fit log lines: the pre-reduction line must say it is correlation clustering (not a groups column) and name the suite
opt-out, and a final line must report how many and which ORIGINAL features were kept."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector

LOGGER = "mlframe.feature_selection.filters.correlated_features"


def _correlated_frame(n=300, seed=0):
    """Build four independent columns plus near-copies of a and b, with a target from a + b."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 4))
    X = pd.DataFrame(base, columns=["a", "b", "c", "d"])
    X["a_copy"] = X["a"] + rng.normal(scale=0.01, size=n)
    X["b_copy"] = X["b"] + rng.normal(scale=0.01, size=n)
    y = (X["a"] + X["b"] > 0).astype(int).to_numpy()
    return X, y


def _messages(caplog):
    """Return the messages logged by the module's logger."""
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


def test_pre_reduction_log_explains_correlation_clusters_and_opt_out(caplog):
    """Pre reduction log explains correlation clusters and opt out."""
    X, y = _correlated_frame()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        CorrelatedFeaturesSelector(SelectKBest(f_classif, k=2), corr_method="pearson").fit(X, y)
    first = next(m for m in _messages(caplog) if "original features ->" in m)
    assert "correlation-cluster pre-reduction (not a groups column)" in first
    assert "6 original features -> 4 clusters" in first
    assert "FeatureSelectionConfig(rfecv={'cluster': {'enable': False}})" in first


def test_fit_logs_kept_original_feature_count_and_names(caplog):
    """Fit logs kept original feature count and names."""
    X, y = _correlated_frame()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        sel = CorrelatedFeaturesSelector(SelectKBest(f_classif, k=2), corr_method="pearson").fit(X, y)
    kept = [X.columns[i] for i in sel.support_]
    summary = [m for m in _messages(caplog) if m.startswith("CorrelatedFeaturesSelector: kept")]
    assert len(summary) == 1
    assert f"kept {len(kept)} of {X.shape[1]} original features" in summary[0]
    assert summary[0].endswith(f"[{', '.join(kept)}]")


def test_fit_logs_kept_features_when_reduction_below_min_reduction(caplog):
    """Fit logs kept features when reduction below min reduction."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("pqrst"))
    y = (X["p"] > 0).astype(int).to_numpy()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        sel = CorrelatedFeaturesSelector(SelectKBest(f_classif, k=2), corr_method="pearson").fit(X, y)
    assert not sel.reduced_
    msgs = _messages(caplog)
    assert any("applied=False" in m and "all original features" in m for m in msgs)
    assert any(m.startswith(f"CorrelatedFeaturesSelector: kept {len(sel.support_)} of 5 original features") for m in msgs)
