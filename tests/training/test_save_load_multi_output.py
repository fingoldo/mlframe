"""Save/load roundtrip tests for multi-output models (Session 5).

Catches joblib-pickle drift in three multilabel dispatch paths:
- Native CB (`loss_function='MultiLogloss'`): CB pickle preserves
  `loss_function`, `classes_` arrays, model state.
- `MultiOutputClassifier` wrapper: sklearn wrapper preserves per-estimator
  `classes_` — CB-inside-wrapper has historical pickle quirks (CB init
  mutates params after first fit).
- `_ChainEnsemble`: `ClassifierChain.order_` is numpy array; cross-version
  pickle compatibility verified by explicit roundtrip.

Each test pattern:
1. Fit a small multilabel model
2. Generate predictions on (X_pred, y_pred_original)
3. joblib.dump → joblib.load
4. Generate predictions on same X_pred
5. Assert np.allclose(original, loaded) — pickle is a pure-data transform,
   must be numerically identical.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from mlframe.training.helpers import (
    _ChainEnsemble,
    _canonical_predict_proba_shape,
)


def _make_multilabel(N=300, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 5)).astype(np.float64)
    logit0 = X[:, 0] - 0.3 * X[:, 1]
    y0 = (logit0 + rng.normal(0, 0.4, N) > 0).astype(np.int8)
    logit1 = 0.5 * y0 + X[:, 2]
    y1 = (logit1 + rng.normal(0, 0.4, N) > 0).astype(np.int8)
    logit2 = 0.5 * y0 + 0.5 * y1 + 0.3 * X[:, 3]
    y2 = (logit2 + rng.normal(0, 0.4, N) > 0.6).astype(np.int8)
    y = np.column_stack([y0, y1, y2])
    # Avoid all-zero rows
    zeros = (y.sum(axis=1) == 0)
    for i in np.where(zeros)[0]:
        y[i, rng.integers(0, K)] = 1
    return X, y


# ---------------------------------------------------------------------------
# Native CatBoost MultiLogloss
# ---------------------------------------------------------------------------


def test_cb_multilogloss_save_load_roundtrip(tmp_path: Path):
    """CatBoost native multilabel: loss_function='MultiLogloss' preserves
    across joblib.dump/load; predictions bit-identical."""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X, y = _make_multilabel()
    X_tr, X_te = X[:240], X[240:]
    y_tr = y[:240]

    model = CatBoostClassifier(
        loss_function="MultiLogloss",
        iterations=15,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr.astype(np.int32))
    original_probs = model.predict_proba(X_te)

    artifact = tmp_path / "cb_multilogloss.joblib"
    joblib.dump(model, artifact)
    loaded = joblib.load(artifact)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_allclose(original_probs, loaded_probs, atol=1e-10)
    # loss_function survives pickle
    assert loaded.get_param("loss_function") == "MultiLogloss"


# ---------------------------------------------------------------------------
# MultiOutputClassifier wrapper
# ---------------------------------------------------------------------------


def test_multioutputclassifier_wrapper_save_load_roundtrip(tmp_path: Path):
    """Wrapper preserves per-estimator classes_ across pickle."""
    X, y = _make_multilabel()
    X_tr, X_te = X[:240], X[240:]
    y_tr = y[:240]

    base = LogisticRegression(max_iter=100, random_state=0)
    moc = MultiOutputClassifier(clone(base))
    moc.fit(X_tr, y_tr)
    original = moc.predict_proba(X_te)
    original_NK = _canonical_predict_proba_shape(original)

    artifact = tmp_path / "moc_wrapper.joblib"
    joblib.dump(moc, artifact)
    loaded = joblib.load(artifact)
    loaded_probs = _canonical_predict_proba_shape(loaded.predict_proba(X_te))

    np.testing.assert_allclose(original_NK, loaded_probs, atol=1e-12)
    # Each estimator's classes_ survived
    assert len(loaded.estimators_) == 3
    for est in loaded.estimators_:
        assert hasattr(est, "classes_")
        assert len(est.classes_) <= 2  # 0 and/or 1


def test_multioutputclassifier_near_constant_label_roundtrip(tmp_path: Path):
    """When a label is ~99% class 0 (realistic rare-positive case), LR fits
    but classes_ still has 2 entries. Roundtrip must preserve prediction
    determinism."""
    X, y = _make_multilabel()
    # Force label 2 to be 99% zeros + 1% ones (rare but LR-fittable).
    y[:, 2] = 0
    y[::100, 2] = 1  # every 100th row gets label 2
    X_tr, X_te = X[:240], X[240:]
    y_tr = y[:240]

    base = LogisticRegression(max_iter=100, random_state=0)
    moc = MultiOutputClassifier(clone(base))
    moc.fit(X_tr, y_tr)
    original = _canonical_predict_proba_shape(moc.predict_proba(X_te))

    artifact = tmp_path / "moc_near_const.joblib"
    joblib.dump(moc, artifact)
    loaded = joblib.load(artifact)
    loaded_probs = _canonical_predict_proba_shape(loaded.predict_proba(X_te))

    np.testing.assert_allclose(original, loaded_probs, atol=1e-12)
    # Column 2 should have very low probabilities (rare positive)
    assert loaded_probs[:, 2].mean() < 0.2, (
        f"Column 2 mean prob {loaded_probs[:, 2].mean():.3f} too high for "
        "99% zeros training target."
    )


# ---------------------------------------------------------------------------
# _ChainEnsemble — ClassifierChain order_ survival
# ---------------------------------------------------------------------------


def test_chain_ensemble_save_load_roundtrip(tmp_path: Path):
    """_ChainEnsemble: chain order_ arrays (sklearn ClassifierChain.order_ is
    numpy array) must survive pickle; predictions bit-identical."""
    X, y = _make_multilabel()
    X_tr, X_te = X[:240], X[240:]
    y_tr = y[:240]

    base = LogisticRegression(max_iter=100, random_state=0)
    chain = _ChainEnsemble(
        clone(base), n_labels=3, n_chains=3,
        seeds=[0, 1, 2], cv=3,
    )
    chain.fit(X_tr, y_tr)
    original = chain.predict_proba(X_te)

    # Capture chain orderings BEFORE pickle
    original_orders = [np.asarray(c.order_).tolist() for c in chain.chains_]

    artifact = tmp_path / "chain_ensemble.joblib"
    joblib.dump(chain, artifact)
    loaded = joblib.load(artifact)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_allclose(original, loaded_probs, atol=1e-12)
    # Order survived
    loaded_orders = [np.asarray(c.order_).tolist() for c in loaded.chains_]
    assert original_orders == loaded_orders, (
        f"Chain orderings drifted across pickle: "
        f"original={original_orders}, loaded={loaded_orders}"
    )


def test_chain_ensemble_by_frequency_order_roundtrip(tmp_path: Path):
    """chain_order_strategy='by_frequency' resolves orders at fit time; they
    must survive roundtrip."""
    X, y = _make_multilabel()
    X_tr = X[:240]
    y_tr = y[:240]

    base = LogisticRegression(max_iter=100, random_state=0)
    chain = _ChainEnsemble(
        clone(base), n_labels=3, n_chains=2, seeds=[0, 1],
        order_strategy="by_frequency", cv=3,
    )
    chain.fit(X_tr, y_tr)

    orig_resolved = [np.asarray(c.order_).tolist() for c in chain.chains_]

    artifact = tmp_path / "chain_by_freq.joblib"
    joblib.dump(chain, artifact)
    loaded = joblib.load(artifact)
    loaded_resolved = [np.asarray(c.order_).tolist() for c in loaded.chains_]

    assert orig_resolved == loaded_resolved
    # by-frequency: all chains share the same ordering (rare-first)
    assert orig_resolved[0] == orig_resolved[1]


# ---------------------------------------------------------------------------
# _PostHocMultiCalibratedModel save/load
# ---------------------------------------------------------------------------


def test_per_class_calibrated_model_save_load_roundtrip(tmp_path: Path):
    """_PerClassIsotonicCalibrator wrapped in _PostHocMultiCalibratedModel
    survives pickle: calibrators dict + is_exclusive flag + target_type."""
    from mlframe.training.trainer import (
        _PerClassIsotonicCalibrator,
        _PostHocMultiCalibratedModel,
    )
    from mlframe.training.configs import TargetTypes

    X, y = _make_multilabel()
    X_tr, X_te = X[:240], X[240:]
    y_tr = y[:240]

    # Fit a base model for per-class probs
    base = MultiOutputClassifier(LogisticRegression(max_iter=100, random_state=0))
    base.fit(X_tr, y_tr)
    probs_tr = _canonical_predict_proba_shape(base.predict_proba(X_tr))

    calibrator = _PerClassIsotonicCalibrator.fit(
        probs_tr, y_tr, TargetTypes.MULTILABEL_CLASSIFICATION,
    )
    wrapped = _PostHocMultiCalibratedModel(
        base, calibrator, TargetTypes.MULTILABEL_CLASSIFICATION,
    )
    original = wrapped.predict_proba(X_te)

    artifact = tmp_path / "calibrated_wrapper.joblib"
    joblib.dump(wrapped, artifact)
    loaded = joblib.load(artifact)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_allclose(original, loaded_probs, atol=1e-12)
    # Target type flag survived (branches wrapped.predict on it)
    assert loaded._target_type == TargetTypes.MULTILABEL_CLASSIFICATION
