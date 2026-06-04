"""Iter-4: serialization roundtrip for the MLP estimators.

Two paths to test:

  1. **stdlib pickle**: must preserve predictions bit-for-bit. This is
     the sklearn-canonical roundtrip contract that frameworks like
     joblib rely on (joblib.dump uses pickle under the hood).

  2. **mlframe production save**: ``save_mlframe_model`` / ``load_mlframe_model``
     from training/io.py. This is what the suite actually uses to
     persist trained models to disk; internally it uses dill but with
     extensive pre-dump stripping (torch.compile state, Lightning
     Trainer back-refs, DataModule/DataLoader bloat).

  3. **direct dill**: ``dill.dumps(estimator)`` now round-trips on its own.
     The LightningModule ``__getstate__`` drops the runtime-only torch.compile
     predict cache (the torch._dynamo ``ConfigModuleInstance`` that previously
     made bare dill raise ``cannot pickle 'ConfigModuleInstance' object``) and
     the CUDA-graph cache, so the estimator no longer depends on the io.py
     strip to serialise.

The tests are PARAMETRIZED across (a) estimator type (regressor vs
classifier-binary vs classifier-multiclass) and (b) serialization
path so a single source of truth covers every flavour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.io import load_mlframe_model, save_mlframe_model
from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _base_params(loss_fn, labels_dtype):
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": loss_fn, "learning_rate": 1e-2},
        "network_params": {
            "nlayers": 1, "first_layer_num_neurons": 16,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32,
            "labels_dtype": labels_dtype,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 2, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "random_state": 0,
    }


# (estimator_factory, data_factory, prediction_method, kind_label)
_CASES = [
    pytest.param(
        lambda: PytorchLightningRegressor(**_base_params(torch.nn.MSELoss(), torch.float32)),
        lambda: _make_data("regression"),
        "predict",
        id="regressor",
    ),
    pytest.param(
        lambda: PytorchLightningClassifier(**_base_params(torch.nn.CrossEntropyLoss(), torch.int64)),
        lambda: _make_data("binary"),
        "predict_proba",
        id="classifier_binary",
    ),
    pytest.param(
        lambda: PytorchLightningClassifier(**_base_params(torch.nn.CrossEntropyLoss(), torch.int64)),
        lambda: _make_data("multiclass"),
        "predict_proba",
        id="classifier_multiclass",
    ),
]


def _make_data(kind: str):
    if kind == "regression":
        X, y = make_regression(n_samples=160, n_features=5, noise=0.1, random_state=0)
        y = y.astype(np.float32)
    elif kind == "binary":
        X, y = make_classification(n_samples=160, n_features=5, n_informative=4,
                                   n_redundant=0, n_classes=2, random_state=0)
        y = y.astype(np.int64)
    elif kind == "multiclass":
        X, y = make_classification(n_samples=180, n_features=6, n_informative=5,
                                   n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                   random_state=0)
        y = y.astype(np.int64)
    else:
        raise ValueError(kind)
    X_tr, X_te, y_tr, _ = train_test_split(X.astype(np.float32), y, test_size=0.3, random_state=0)
    return X_tr, X_te, y_tr


@pytest.mark.parametrize("estimator_factory, data_factory, prediction_method", _CASES)
def test_stdlib_pickle_preserves_predictions(estimator_factory, data_factory, prediction_method):
    """stdlib pickle roundtrip must produce bit-identical predictions
    across regressor / binary classifier / multiclass classifier."""
    import pickle
    X_tr, X_te, y_tr = data_factory()
    est = estimator_factory()
    est.fit(X_tr, y_tr)
    pred_before = getattr(est, prediction_method)(X_te)

    buf = pickle.dumps(est)
    est2 = pickle.loads(buf)
    pred_after = getattr(est2, prediction_method)(X_te)

    np.testing.assert_array_equal(pred_before, pred_after)


@pytest.mark.parametrize("estimator_factory, data_factory, prediction_method", _CASES)
def test_mlframe_save_load_preserves_predictions(
    estimator_factory, data_factory, prediction_method, tmp_path,
):
    """Production save/load path via ``save_mlframe_model`` /
    ``load_mlframe_model`` (handles torch.compile bloat strip + Lightning
    Trainer/DataModule strip internally per io.py:544-594). Roundtrip
    must produce bit-identical predictions."""
    X_tr, X_te, y_tr = data_factory()
    est = estimator_factory()
    est.fit(X_tr, y_tr)
    pred_before = getattr(est, prediction_method)(X_te)

    save_path = str(tmp_path / "estimator.bundle")
    save_mlframe_model(est, save_path, verbose=0)
    est2 = load_mlframe_model(save_path)
    pred_after = getattr(est2, prediction_method)(X_te)

    np.testing.assert_array_equal(pred_before, pred_after)


def test_classifier_pickle_preserves_label_encoder_and_classes():
    """Non-dense labels: classes_ + _label_encoder must survive pickle
    (sklearn convention: ``predict()`` returns ORIGINAL labels)."""
    import pickle
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 5)).astype(np.float32)
    y_dense = (X[:, 0] > 0).astype(np.int64)
    y_nondense = np.where(y_dense == 0, 10, 20).astype(np.int64)
    X_tr, X_te, y_tr, _ = train_test_split(X, y_nondense, test_size=0.3, random_state=0)

    clf = PytorchLightningClassifier(**_base_params(torch.nn.CrossEntropyLoss(), torch.int64))
    clf.fit(X_tr, y_tr)
    assert set(clf.classes_.tolist()) == {10, 20}

    clf2 = pickle.loads(pickle.dumps(clf))
    assert hasattr(clf2, "_label_encoder")
    assert clf2._label_encoder is not None
    assert set(clf2.classes_.tolist()) == {10, 20}
    preds = clf2.predict(X_te)
    assert set(np.asarray(preds).tolist()).issubset({10, 20})


def test_direct_dill_dumps_roundtrips_via_getstate_strip():
    """Direct ``dill.dumps(estimator)`` round-trips cleanly.

    Historically this raised ``cannot pickle 'ConfigModuleInstance' object``
    because dill walked into the LightningModule's live ``_compiled_predict_fn``
    (a torch._dynamo ``ConfigModuleInstance``) and the CUDA-graph predict cache.
    The module's ``__getstate__`` (``_flat_torch_module.py``) now drops both
    runtime-only fast-path caches on serialise, so the whole estimator pickles
    with bare dill -- no longer dependent on the ``save_mlframe_model`` strip.
    This test PINS the round-trip so a future change that re-introduces a live
    non-picklable cache into the serialised state surfaces visibly.
    """
    pytest.importorskip("dill")
    import dill
    X_tr, X_te, y_tr = _make_data("regression")
    reg = PytorchLightningRegressor(**_base_params(torch.nn.MSELoss(), torch.float32))
    reg.fit(X_tr, y_tr)
    pred_before = reg.predict(X_te)

    reg2 = dill.loads(dill.dumps(reg))
    pred_after = reg2.predict(X_te)
    np.testing.assert_array_equal(pred_before, pred_after)
