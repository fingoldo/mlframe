"""Regression test for the multilabel-MLP BCE dispatch in PytorchLightningClassifier.

Pre-fix (commit ranges spanning 2026-05-20 fuzz combo c0030):
``helpers.py:751`` defaults ``loss_fn = F.cross_entropy`` for the classifier
path with no branch for multilabel. When the target is (N, K) with K >= 2, the
estimator detected multilabel and built an MLP with K outputs — but CE got
fed (N, K) Long labels and raised ``RuntimeError: The size of tensor a (K)
must match the size of tensor b (N*K) at non-singleton dimension 1``.

Post-fix: ``_fit_common`` switches the per-fit ``loss_fn`` to
``F.binary_cross_entropy_with_logits``, tags ``task_type="multilabel"`` so
``predict_step`` applies sigmoid (not softmax), and bumps ``labels_dtype`` to
float32 at the datamodule layer (BCE refuses Long). The estimator's
``self.model_params`` / ``self.datamodule_params`` are NOT mutated — only the
local per-fit copies — so subsequent fits on a SINGLE-LABEL target work
unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning.pytorch")

from mlframe.training.neural.base import PytorchLightningClassifier  # noqa: E402
from mlframe.training.neural.data import TorchDataModule  # noqa: E402
from mlframe.training.neural.flat import MLPTorchModel  # noqa: E402


def _build_classifier(n_features: int, max_epochs: int = 2) -> PytorchLightningClassifier:
    """Mirror the helpers.py classifier-config defaults so this test exercises
    the exact path the fuzz suite hits."""
    import torch.nn.functional as F

    model_params = dict(
        loss_fn=F.cross_entropy,  # the broken default; the fix overrides this for multilabel
        learning_rate=3e-3,
        l1_alpha=0.0,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={},
        lr_scheduler=None,
        lr_scheduler_kwargs={},
    )
    datamodule_params = dict(
        features_dtype=torch.float32,
        labels_dtype=torch.int64,  # broken default for multilabel; fix overrides at the datamodule layer
        dataloader_params=dict(batch_size=32, num_workers=0),
    )
    trainer_params = dict(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        enable_model_summary=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    network_params = dict(
        nlayers=2,
        first_layer_num_neurons=16,
        min_layer_neurons=4,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_layernorm=False,
        use_batchnorm=False,
        verbose=0,
    )
    return PytorchLightningClassifier(
        model_class=MLPTorchModel,
        model_params=model_params,
        network_params=network_params,
        datamodule_class=TorchDataModule,
        datamodule_params=datamodule_params,
        trainer_params=trainer_params,
        early_stopping_rounds=10,
    )


def test_multilabel_mlp_fits_with_bce_loss():
    """A 3-label multilabel target must train end-to-end without shape errors.

    This is the exact failure shape from fuzz combo c0030_6a14550a:
    PytorchLightningClassifier + MLP + (N, K=3) multilabel target raised
    RuntimeError ``The size of tensor a (3) must match the size of tensor
    b (65536)`` because the CE-default loss got int64 (N, K) labels.
    """
    rng = np.random.default_rng(20260520)
    n, p, k = 200, 8, 3
    X = rng.standard_normal((n, p)).astype(np.float32)
    # Y of shape (N, K) with independent binary labels (multilabel).
    y = (rng.random((n, k)) > 0.5).astype(np.int64)

    clf = _build_classifier(n_features=p, max_epochs=2)
    # This used to raise inside Trainer.fit -> training_step ->
    # cross_entropy(logits=(B, K), labels=(B, K)). Must train cleanly now.
    clf.fit(X, y)
    # Sanity: estimator correctly tagged itself.
    assert clf._is_multilabel is True
    assert clf.n_labels_ == k
    # The shipped loss must be BCE; CE would still throw at fit time.
    assert clf.model.loss_fn is torch.nn.functional.binary_cross_entropy_with_logits
    assert clf.model.task_type == "multilabel"


def test_single_label_classification_still_uses_cross_entropy():
    """The fix must NOT affect single-label classifiers. Sklearn-1D target
    (N,) goes through the unchanged CE path; multi-class 1D target same."""
    import torch.nn.functional as F

    rng = np.random.default_rng(20260520)
    n, p = 200, 8
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = rng.integers(0, 3, size=n).astype(np.int64)  # 1-D multi-class, 3 classes

    clf = _build_classifier(n_features=p, max_epochs=2)
    clf.fit(X, y)
    assert clf._is_multilabel is False
    # loss_fn unchanged at CE; task_type stays None (multiclass default).
    assert clf.model.loss_fn is F.cross_entropy
    assert clf.model.task_type is None


def test_estimator_params_not_mutated_by_multilabel_fit():
    """The fix must NOT mutate self.model_params / self.datamodule_params —
    sklearn clone() relies on constructor params being untouched, and a
    follow-up fit on single-label data must NOT inherit the BCE override."""
    import torch.nn.functional as F

    rng = np.random.default_rng(20260520)
    n, p, k = 80, 6, 2
    X_ml = rng.standard_normal((n, p)).astype(np.float32)
    y_ml = (rng.random((n, k)) > 0.5).astype(np.int64)

    clf = _build_classifier(n_features=p, max_epochs=1)
    pre_loss_fn = clf.model_params["loss_fn"]
    pre_labels_dtype = clf.datamodule_params["labels_dtype"]
    clf.fit(X_ml, y_ml)
    # The original constructor params are untouched.
    assert clf.model_params["loss_fn"] is pre_loss_fn, "model_params['loss_fn'] mutated"
    assert clf.datamodule_params["labels_dtype"] is pre_labels_dtype, "datamodule_params['labels_dtype'] mutated"
    assert pre_loss_fn is F.cross_entropy
    assert pre_labels_dtype is torch.int64
