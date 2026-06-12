"""Regression sensor for D1 P2 #8.

BorutaShap's ``explain()`` must use ONLY the train slice as the SHAP
background dataset (``self.X_boruta`` is built from ``self.X``, which is set
in ``fit()`` from the caller-supplied X). Mixing in val/test rows would
let the TreeExplainer interpolate against held-out distribution and
inflate borderline-feature importances.

The guard:
- Asserts ``self.X_boruta`` row count matches ``self.X`` row count
  when sample=False.
- Emits an INFO line recording (n_train, n_basis, sampled) so callers
  see the train-background discipline in the log.

This sensor patches ``shap.TreeExplainer.shap_values`` with a captor and
verifies the call basis has the same row count as the train X handed to fit().
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def boruta_train_basis_captor(monkeypatch):
    """Patch shap.TreeExplainer.shap_values to capture the basis frame's shape."""
    import shap

    captured = {}
    original = shap.TreeExplainer.shap_values

    def _capturing_shap_values(self, X, *args, **kwargs):
        captured["n_rows_basis"] = int(X.shape[0])
        captured["n_cols_basis"] = int(X.shape[1])
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(shap.TreeExplainer, "shap_values", _capturing_shap_values)
    return captured


def test_d1_p2_8_shap_explainer_uses_train_only_basis(boruta_train_basis_captor, caplog):
    """Drive ``explain()`` directly with a hand-set ``self.X_boruta`` so the
    test does not depend on the multi-trial inner loop's RNG / shape stability.

    The guard under audit lives inside ``explain()``: it asserts
    ``self.X_boruta`` row count matches ``self.X`` row count and logs an
    INFO line with both counts. Both behaviours are exercised here.
    """
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X_full, y_full = make_classification(
        n_samples=80, n_features=4, n_informative=2,
        n_redundant=0, random_state=42,
    )
    X_train_df = pd.DataFrame(X_full, columns=[f"f{i}" for i in range(4)])
    y_train = pd.Series(y_full)

    bs = BorutaShap(
        model=RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1),
        importance_measure="shap",
        classification=True,
        n_trials=1,
        random_state=42,
        verbose=False,
        sample=False,
    )
    # Manually set the minimum state explain() requires: a fitted self.model,
    # self.X (train), self.X_boruta = [self.X | shadow], self.y, and the
    # classification flag (set in __init__).
    rng = np.random.default_rng(42)
    shadow = pd.DataFrame(
        rng.permutation(X_train_df.values, axis=0),
        columns=["shadow_" + c for c in X_train_df.columns],
        index=X_train_df.index,
    )
    bs.X = X_train_df.copy()
    bs.X_boruta = pd.concat([X_train_df, shadow], axis=1)
    bs.y = y_train
    bs.model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    bs.model.fit(bs.X_boruta, bs.y)

    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.boruta_shap._fit_explain"):
        bs.explain()

    assert "n_rows_basis" in boruta_train_basis_captor, "shap_values should have been called"
    # 80 train rows; basis = X_boruta has same n_rows
    assert boruta_train_basis_captor["n_rows_basis"] == 80, (
        f"Expected SHAP basis n_rows={80} (train-only); got {boruta_train_basis_captor['n_rows_basis']}"
    )
    info_msgs = [r.message for r in caplog.records if r.levelno >= logging.INFO]
    assert any("train background" in m and "n_train=80" in m for m in info_msgs), (
        f"Expected INFO log mentioning train background + n_train; got: {info_msgs}"
    )


def test_d1_p2_8_shap_explainer_assertion_fires_on_size_mismatch():
    """If a bad refactor lets val rows into ``X_boruta``, the guard MUST raise."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X_full, y_full = make_classification(
        n_samples=80, n_features=4, n_informative=2,
        n_redundant=0, random_state=42,
    )
    X_train_df = pd.DataFrame(X_full[:60], columns=[f"f{i}" for i in range(4)])
    # Bad: X_boruta contains all 80 rows but X only has 60 (=> val leaked in)
    X_bad_boruta = pd.DataFrame(
        np.hstack([X_full, X_full]),
        columns=[f"f{i}" for i in range(4)] + [f"shadow_f{i}" for i in range(4)],
    )
    bs = BorutaShap(
        model=RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1),
        importance_measure="shap",
        classification=True,
        n_trials=1,
        random_state=42,
        verbose=False,
        sample=False,
    )
    bs.X = X_train_df
    bs.X_boruta = X_bad_boruta
    bs.y = pd.Series(y_full[:60])
    bs.model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    bs.model.fit(X_bad_boruta, pd.Series(y_full))
    with pytest.raises(AssertionError, match="SHAP background row count"):
        bs.explain()
