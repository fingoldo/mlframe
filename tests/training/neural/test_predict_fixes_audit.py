"""Regression tests for three neural predict()-path audit fixes.

#7  predict(device=, precision=) were applied only in the live-trainer branch of
    _predict_raw, but self.trainer is reset to None after fit / after every
    predict, so the normal post-fit path took the other branch and silently
    dropped them. They now apply in both paths.
#8  setup_predict(batch_size=) was shadowed by dataloader_params['batch_size']
    (the suite seeds it 'auto'), so the predict-batch override was discarded.
    setup_predict now mirrors the override into dataloader_params.
#9  multilabel predict() fell through to argmax -> (N,) labels; it must return
    the (N, K) per-label 0/1 indicator matrix.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural import (  # noqa: E402
    MLPTorchModel,
    PytorchLightningClassifier,
    TorchDataModule,
)


def _classifier(max_epochs=1):
    return PytorchLightningClassifier(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.CrossEntropyLoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 1, "first_layer_num_neurons": 8, "dropout_prob": 0.0,
                        "activation_function": torch.nn.ReLU},
        datamodule_class=TorchDataModule,
        datamodule_params={"read_fcn": None, "data_placement_device": None,
                           "features_dtype": torch.float32, "labels_dtype": torch.int64,
                           "dataloader_params": {"batch_size": 32, "num_workers": 0}},
        trainer_params={"max_epochs": max_epochs, "enable_model_summary": False,
                        "default_root_dir": None, "log_every_n_steps": 1, "devices": 1,
                        "logger": False, "accelerator": "cpu"},
    )


def test_multilabel_predict_returns_indicator_matrix_not_argmax():
    """#9: with _is_multilabel set, predict() returns the (N, K) 0/1 matrix from
    the per-label sigmoid raw output -- not a single argmax label per row."""
    clf = _classifier()
    clf._is_multilabel = True
    clf._binary_sigmoid_head = False
    clf._label_encoder = None
    clf.classes_ = None
    raw = np.array([[0.9, 0.1, 0.8], [0.2, 0.95, 0.3], [0.4, 0.6, 0.49]])
    clf._predict_raw = lambda X, **kw: raw  # bypass the torch forward pass

    out = clf.predict(np.zeros((3, 5), dtype=np.float32))
    assert out.shape == (3, 3), f"multilabel predict must be (N, K); got {out.shape}"
    np.testing.assert_array_equal(out, (raw >= 0.5).astype(np.int64))


def test_setup_predict_batch_size_overrides_dataloader_auto():
    """#8: setup_predict(batch_size=N) must win over a dataloader_params
    'batch_size' default (the suite seeds 'auto'), so the predict dataloader
    uses the resolved predict batch instead of the train resolver."""
    dm = TorchDataModule(
        read_fcn=None, data_placement_device=None,
        features_dtype=torch.float32, labels_dtype=torch.int64,
        dataloader_params={"batch_size": "auto", "num_workers": 0},
    )
    X = np.random.default_rng(0).normal(size=(40, 5)).astype(np.float32)
    dm.setup_predict(X, batch_size=128)
    # Pre-fix dataloader_params['batch_size'] stayed 'auto' and shadowed self.batch_size.
    assert dm.dataloader_params.get("batch_size") == 128, (
        f"predict batch override must be mirrored into dataloader_params; "
        f"got {dm.dataloader_params.get('batch_size')!r}"
    )


def test_predict_device_arg_honored_in_post_fit_path():
    """#7: after fit (self.trainer reset to None), predict(device='cpu') must
    still route the trainer to the cpu accelerator -- pre-fix the device arg was
    only read in the unreachable live-trainer branch."""
    import mlframe.training.neural.base as _nb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    clf = _classifier(max_epochs=1)
    clf.fit(X, y)
    # After fit, self.trainer is None -> the post-fit branch builds trainer_params.

    captured = {}
    _RealTrainer = _nb.L.Trainer

    def _spy_trainer(**kwargs):
        captured.update(kwargs)
        return _RealTrainer(**kwargs)

    import unittest.mock as _mock
    with _mock.patch.object(_nb.L, "Trainer", _spy_trainer):
        clf.predict(X, device="cpu")

    assert captured.get("accelerator") == "cpu", (
        f"predict(device='cpu') must set accelerator='cpu' in the post-fit path; "
        f"captured trainer_params accelerator={captured.get('accelerator')!r}"
    )
