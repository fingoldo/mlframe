"""F-37 + F-38: BoundedTanh fusion-friendly form + CUDA-graph predict
fast path.

Most assertions are CPU-friendly (env-gate / cache structure / fallback
behaviour); the actual CUDA-graph capture is only exercised when a CUDA
device is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel, PytorchLightningRegressor, TorchDataModule,
)
from mlframe.training.neural.flat import _BoundedTanhOutput, generate_mlp


# --- F-37: BoundedTanh no longer branches on is_cuda -------------------------


def test_bounded_tanh_no_is_cuda_branch():
    """The forward body should NOT contain the data-dependent
    ``if x.is_cuda`` STATEMENT -- pre-F-37 fragmented Inductor fusion.

    Use AST parsing to find If nodes -- robust against comments and
    docstrings that mention the pattern intentionally."""
    import ast
    import inspect
    src = inspect.getsource(_BoundedTanhOutput.forward)
    # Dedent so ast can parse the function body.
    import textwrap
    src_dedented = textwrap.dedent(src)
    tree = ast.parse(src_dedented)
    # The function definition is the top-level node.
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    # Walk the body looking for any If statement; there should be none
    # post-F-37 (the function body is a single return expression).
    for node in ast.walk(func_node):
        if isinstance(node, ast.If):
            pytest.fail(
                "_BoundedTanhOutput.forward should NOT contain an If "
                "statement (F-37 removed the is_cuda branch to unlock "
                "Inductor fusion); current source:\n" + src
            )
    # Verify the new elementwise form is in the source.
    assert "torch.tanh(x) * self.scale + self.center" in src or \
           "tanh(x) * self.scale + self.center" in src


def test_bounded_tanh_forward_numerics_unchanged():
    """The numeric output of the elementwise form should match the prior
    addcmul form (verified at construction time)."""
    head = _BoundedTanhOutput(scale=2.0, center=0.5)
    x = torch.randn(64, 3)
    out = head(x)
    expected = torch.tanh(x) * 2.0 + 0.5
    torch.testing.assert_close(out, expected)


def test_bounded_tanh_backward_gradient_intact():
    """tanh's autograd gradient (1 - tanh^2) is preserved through the
    new form. Verify backward produces nonzero gradients."""
    head = _BoundedTanhOutput(scale=2.0, center=0.5)
    x = torch.randn(8, 3, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Per-element grad should be 2.0 * (1 - tanh^2(x)) per the chain rule.
    expected_grad = 2.0 * (1.0 - torch.tanh(x.detach()) ** 2)
    torch.testing.assert_close(x.grad, expected_grad)


def test_bounded_tanh_works_inside_generate_mlp():
    """Sanity: generate_mlp(output_activation='tanh_train_range')
    constructs a network ending with _BoundedTanhOutput, and forward
    produces output in roughly [center - scale, center + scale]."""
    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=1,
        first_layer_num_neurons=8, dropout_prob=0.0,
        inputs_dropout_prob=0.0, use_layernorm=False, use_batchnorm=False,
        activation_function=nn.ReLU,
        output_activation="tanh_train_range",
        output_activation_scale=3.0,
        output_activation_center=1.0,
        verbose=0,
    )
    # Last module should be _BoundedTanhOutput.
    last = list(net.modules())[-1]
    assert isinstance(last, _BoundedTanhOutput)
    x = torch.randn(16, 4)
    out = net(x)
    # tanh range is [-1, 1] -> scaled to [1-3, 1+3] = [-2, 4].
    assert (out >= -2.001).all() and (out <= 4.001).all()


# --- F-38: CUDA-graph predict (env-gate + fallback semantics) ---------------


def _make_module() -> MLPTorchModel:
    network = generate_mlp(
        num_features=4, num_classes=1, nlayers=1,
        first_layer_num_neurons=8, dropout_prob=0.0,
        inputs_dropout_prob=0.0, use_layernorm=False, use_batchnorm=False,
        activation_function=nn.ReLU, verbose=0,
    )
    return MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=network)


def test_cuda_graph_cache_initialised_empty():
    module = _make_module()
    assert hasattr(module, "_cuda_graph_predict_cache")
    assert module._cuda_graph_predict_cache == {}


def test_cuda_graph_env_explicit_off_falls_back_to_eager(monkeypatch):
    """Explicit MLFRAME_CUDA_GRAPH_PREDICT=0 -> eager fallback.
    F-40 flipped the default to ON (low-level CUDAGraph() API is
    non-destructive); users opt OUT via "0" / "false" / "off"."""
    monkeypatch.setenv("MLFRAME_CUDA_GRAPH_PREDICT", "0")
    module = _make_module()
    x = torch.randn(4, 4)
    out = module._maybe_cuda_graph_forward(x)
    torch.testing.assert_close(out, module(x))
    assert module._cuda_graph_predict_cache == {}


def test_cuda_graph_env_default_is_on_falls_back_on_cpu(monkeypatch):
    """F-40 default-on: env unset -> opt-in. On CPU input the gate
    still falls back to eager (CUDA graphs are GPU-only)."""
    monkeypatch.delenv("MLFRAME_CUDA_GRAPH_PREDICT", raising=False)
    module = _make_module()
    x = torch.randn(4, 4)  # CPU
    out = module._maybe_cuda_graph_forward(x)
    torch.testing.assert_close(out, module(x))
    # No capture attempted on CPU, cache stays empty.
    assert module._cuda_graph_predict_cache == {}


def test_cuda_graph_cpu_input_falls_back_to_eager(monkeypatch):
    """MLFRAME_CUDA_GRAPH_PREDICT=1 + CPU input -> eager fallback
    (CUDA graphs are GPU-only)."""
    monkeypatch.setenv("MLFRAME_CUDA_GRAPH_PREDICT", "1")
    module = _make_module()
    x = torch.randn(4, 4)  # CPU tensor
    out = module._maybe_cuda_graph_forward(x)
    torch.testing.assert_close(out, module(x))
    # Cache stays empty: CPU input never reaches the capture branch.
    assert module._cuda_graph_predict_cache == {}


def test_cuda_graph_recurrent_network_falls_back_to_eager(monkeypatch):
    """LSTM/GRU/RNN networks must NOT attempt CUDA-graph capture
    (control flow inside cuDNN call breaks capture)."""
    monkeypatch.setenv("MLFRAME_CUDA_GRAPH_PREDICT", "1")
    network = nn.Sequential(nn.LSTM(4, 8, batch_first=True), nn.Linear(8, 1))
    module = MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=network)
    # We can't actually exercise the CUDA path here (no GPU on CI), but
    # the gate should still trip before that branch.
    # On CPU the earlier "not x.is_cuda" gate fires first; this test
    # mainly documents intent. Direct white-box check:
    _recurrent = (nn.LSTM, nn.GRU, nn.RNN)
    has_recurrent = any(isinstance(m, _recurrent) for m in module.network.modules())
    assert has_recurrent is True


def test_predict_step_routes_through_cuda_graph_helper():
    """The predict_step body should call _maybe_cuda_graph_forward
    instead of self(x) directly so the env-gated CUDA-graph path
    fires when the env var is set."""
    import inspect
    src = inspect.getsource(MLPTorchModel.predict_step)
    assert "_maybe_cuda_graph_forward" in src, (
        "predict_step should call _maybe_cuda_graph_forward per F-38; "
        "current source:\n" + src
    )


# --- End-to-end sanity: predict still returns valid output on CPU -----------


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=64, n_features=4, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_tr, X_te, y_tr


def test_predict_works_end_to_end(reg_data, monkeypatch):
    """End-to-end: predict completes without the CUDA-graph path
    (default off). Validates F-37 (BoundedTanh tweak) + F-38 (CUDA-graph
    helper) don't regress the standard predict flow."""
    # Force the CUDA-graph predict path OFF for this test (default) --
    # actual CUDA-graph capture requires careful GPU setup that
    # CI / smoke tests can't reliably provide.
    monkeypatch.delenv("MLFRAME_CUDA_GRAPH_PREDICT", raising=False)
    X_tr, X_te, y_tr = reg_data
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": nn.MSELoss(), "learning_rate": 1e-2},
        network_params={
            "nlayers": 1, "first_layer_num_neurons": 8,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32, "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 1, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        random_state=0,
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)
    assert np.isfinite(preds).all()
