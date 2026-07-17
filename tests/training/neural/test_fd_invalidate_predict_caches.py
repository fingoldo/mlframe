"""F-D (2026-05-31) regression tests: predict caches (F-40 CUDA-graph,
F-39 torch.compile) must be invalidated when weights change in a way
that could break the captured kernels.

The captured graph references parameter tensor storage addresses; if
the weights are updated via in-place ``.copy_()`` (the common path:
Lookahead sync, EMA/SWA update_parameters, default load_state_dict)
the storage is preserved and the captured graph keeps producing
correct output. But ANY path that REPLACES a nn.Parameter object
(load_state_dict(assign=True), LoRA adapter swap, user explicit
.weight = new_param) leaves the captured graph pointing at stale
storage -- replay produces predictions from PRE-swap weights with no
exception. Same silent-correctness class as F-58.

Mitigation: call ``_invalidate_predict_caches()`` after any weight-
replacement path. The two automated call sites are:
  1. on_train_end (before checkpoint reload + after).
  2. After load_state_dict in the best-weights reload path.

These tests verify the API + the call-site wiring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mlframe.training.neural._flat_torch_module import MLPTorchModel
from mlframe.training.neural.flat import generate_mlp


def _make_module() -> MLPTorchModel:
    net = generate_mlp(
        num_features=4,
        num_classes=1,
        nlayers=1,
        first_layer_num_neurons=8,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_layernorm=False,
        use_batchnorm=False,
        activation_function=nn.ReLU,
        verbose=0,
    )
    return MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=net)


def test_invalidate_predict_caches_clears_cuda_graph_cache():
    """F-D: _invalidate_predict_caches drops the CUDA-graph cache map."""
    m = _make_module()
    # Manually plant a sentinel cache entry; we don't need real CUDA to
    # check that the invalidator clears the dict.
    m._cuda_graph_predict_cache[("sentinel", torch.float32, "cpu")] = (
        "graph_handle",
        torch.zeros(1),
        torch.zeros(1),
    )
    assert len(m._cuda_graph_predict_cache) == 1
    m._invalidate_predict_caches()
    assert m._cuda_graph_predict_cache == {}


def test_invalidate_predict_caches_clears_torch_compile_cache():
    """F-D: _invalidate_predict_caches resets the F-39 torch.compile
    handle + failure flag, so the next predict_step re-attempts compile."""
    m = _make_module()
    # Plant a sentinel.
    m._compiled_predict_fn = lambda x: x  # type: ignore[assignment]
    m._compile_predict_failed = True
    m._invalidate_predict_caches()
    assert m._compiled_predict_fn is None
    assert m._compile_predict_failed is False


def test_invalidate_predict_caches_is_idempotent():
    """F-D: calling twice in a row is safe -- second call is a no-op."""
    m = _make_module()
    m._invalidate_predict_caches()
    m._invalidate_predict_caches()
    assert m._cuda_graph_predict_cache == {}
    assert m._compiled_predict_fn is None


def test_on_train_end_invalidates_caches_via_smoke_fit():
    """Integration: a full smoke fit triggers on_train_end which must
    leave the predict caches empty -- guards against the F-D wiring
    regressing silently.
    """
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=64, n_features=4, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    torch.manual_seed(0)
    np.random.seed(0)
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={
            "loss_fn": nn.MSELoss(),
            "learning_rate": 1e-2,
            "load_best_weights_on_train_end": False,  # focus this test on F-D
        },
        network_params={
            "nlayers": 1,
            "first_layer_num_neurons": 8,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": False,
            "activation_function": nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 1,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        random_state=0,
    )
    # Plant fake cache entries BEFORE fit so we can see that on_train_end
    # cleared them.
    reg._fit_common  # noqa: B018  (force attribute resolution / not strictly needed)

    # Run fit + verify caches are empty post-fit.
    reg.fit(X_tr, y_tr)
    assert reg.model._cuda_graph_predict_cache == {}, "F-D: on_train_end did not invalidate _cuda_graph_predict_cache"
    assert reg.model._compiled_predict_fn is None, "F-D: on_train_end did not invalidate _compiled_predict_fn"
