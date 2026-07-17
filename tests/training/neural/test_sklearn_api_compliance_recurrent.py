"""sklearn-contract regression tests for the recurrent wrappers (SK4a/SK4b).

SK4a: ``RecurrentClassifierWrapper`` must set ``classes_`` in fit and map
predict / predict_proba columns through it, so a label set like ``[2, 5, 9]``
round-trips to the ORIGINAL labels (not positional 0/1/2).

SK4b: ``config`` is stored VERBATIM (None stays None across clone) and fit
mutates only a fit-local deepcopy, never ``self.config`` nor the shared
module-level default ``RecurrentConfig()``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")

from mlframe.training.neural.recurrent_dataset_helpers import (
    RecurrentClassifierWrapper,
    RecurrentRegressorWrapper,
)
from mlframe.training.neural._recurrent_config import RecurrentConfig, InputMode


def _fast_features_only_config(num_classes: int = 3, monitor: str = "val_loss") -> RecurrentConfig:
    return RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        max_epochs=1,
        batch_size=16,
        hidden_size=8,
        num_layers=1,
        mlp_hidden_sizes=(8,),
        n_heads=1,
        accelerator="cpu",
        precision="32-true",
        num_workers=0,
        use_stratified_sampler=False,
        scale_features=False,
        num_classes=num_classes,
        early_stopping_monitor=monitor,
    )


def test_sk4a_classifier_predict_returns_original_labels():
    rng = np.random.default_rng(0)
    n = 30
    X = rng.standard_normal((n, 4)).astype(np.float32)
    # Non-0..k-1 label set: a raw positional argmax would mislabel these.
    labels = np.array([2, 5, 9])[rng.integers(0, 3, size=n)]

    est = RecurrentClassifierWrapper(config=_fast_features_only_config(num_classes=3))
    est.fit(features=X, labels=labels)

    assert hasattr(est, "classes_")
    np.testing.assert_array_equal(est.classes_, np.array([2, 5, 9]))

    proba = est.predict_proba(features=X)
    assert proba.shape == (n, 3)

    preds = est.predict(features=X)
    # Every prediction must be one of the original labels, never a positional 0/1/2 index.
    assert set(np.unique(preds)).issubset({2, 5, 9})
    # predict must equal classes_ indexed by the predict_proba argmax (column alignment to classes_).
    np.testing.assert_array_equal(preds, est.classes_[proba.argmax(axis=1)])


@pytest.mark.parametrize("cls", [RecurrentClassifierWrapper, RecurrentRegressorWrapper])
def test_sk4b_clone_config_none_round_trips(cls):
    est = cls()  # config defaults to None
    assert est.config is None
    cloned = clone(est)
    assert cloned.config is None


def test_sk4b_fit_does_not_mutate_passed_config_or_shared_default():
    # Snapshot a fresh default to confirm the shared module-level default is never mutated by a fit.
    reference = copy.deepcopy(RecurrentConfig())

    # num_classes=2 but a 3-column multilabel y: pre-fix code did ``self.config.num_classes = 3`` IN PLACE, mutating the caller's object.
    cfg = RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        max_epochs=1,
        batch_size=16,
        hidden_size=8,
        num_layers=1,
        mlp_hidden_sizes=(8,),
        n_heads=1,
        accelerator="cpu",
        precision="32-true",
        num_workers=0,
        use_stratified_sampler=False,
        scale_features=False,
        num_classes=2,
    )
    cfg_snapshot = copy.deepcopy(cfg)

    rng = np.random.default_rng(1)
    X = rng.standard_normal((24, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=(24, 3)).astype(np.float32)  # multilabel

    est = RecurrentClassifierWrapper(config=cfg)
    est.fit(features=X, labels=y)

    # The caller's config object must be untouched (the num_classes override lands on the fit-local copy).
    assert cfg == cfg_snapshot
    # A brand-new default RecurrentConfig() must equal the reference: the shared default was never mutated.
    assert RecurrentConfig() == reference


def test_sk4b_regressor_monitor_override_does_not_mutate_passed_config():
    cfg = _fast_features_only_config(num_classes=1, monitor="val_auprc")
    cfg_snapshot = copy.deepcopy(cfg)

    rng = np.random.default_rng(2)
    X = rng.standard_normal((24, 4)).astype(np.float32)
    y = rng.standard_normal(24).astype(np.float32)

    est = RecurrentRegressorWrapper(config=cfg)
    est.fit(features=X, labels=y)

    # The val_auprc -> val_loss redirect must land on the fit-local copy, never the caller's config.
    assert cfg.early_stopping_monitor == "val_auprc"
    assert cfg == cfg_snapshot
