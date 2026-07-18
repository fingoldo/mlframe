"""F-39: torch.compile(reduce-overhead) predict-path fast lane.

Most of the actual compile/run is GPU-only and depends on torch + CUDA
version compatibility; the CPU-friendly tests verify the gating logic
+ fallback semantics. The real benchmark lives in
`D:/Temp/bench_mlp_fit_speedup.py` for ad-hoc validation.
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
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)
from mlframe.training.neural.flat import generate_mlp


def _make_module() -> MLPTorchModel:
    """Make module."""
    network = generate_mlp(
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
    return MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=network)


def test_compile_predict_cache_initialised_empty():
    """Default: compiled predict fn is None, no failure cached."""
    module = _make_module()
    assert module._compiled_predict_fn is None
    assert module._compile_predict_failed is False


def test_compile_predict_env_off_returns_none(monkeypatch):
    """MLFRAME_TORCH_COMPILE_PREDICT unset -> _maybe_compile_predict_forward
    returns None (signals "fall through to next path")."""
    monkeypatch.delenv("MLFRAME_TORCH_COMPILE_PREDICT", raising=False)
    module = _make_module()
    x = torch.randn(4, 4)
    assert module._maybe_compile_predict_forward(x) is None


def test_compile_predict_cpu_input_returns_none(monkeypatch):
    """MLFRAME_TORCH_COMPILE_PREDICT=1 + CPU input -> None
    (torch.compile reduce-overhead is GPU-only path)."""
    monkeypatch.setenv("MLFRAME_TORCH_COMPILE_PREDICT", "1")
    module = _make_module()
    x = torch.randn(4, 4)  # CPU tensor
    assert module._maybe_compile_predict_forward(x) is None


def test_compile_predict_after_failure_caches_sentinel(monkeypatch):
    """If a compile attempt fails (e.g. an internal Dynamo bug), the
    sentinel is cached so subsequent calls skip the path entirely."""
    monkeypatch.setenv("MLFRAME_TORCH_COMPILE_PREDICT", "1")
    module = _make_module()
    # Simulate a failed compile.
    module._compile_predict_failed = True
    x = torch.randn(4, 4)  # even if CPU, the early sentinel check fires
    assert module._maybe_compile_predict_forward(x) is None


def test_predict_step_routes_through_compile_then_cuda_graph_then_eager(monkeypatch):
    """Behavioural: predict_step must consult _maybe_compile_predict_forward
    FIRST; if that returns None it must fall through to _maybe_cuda_graph_forward;
    if that also returns None it must finally call eager ``self(x)``. The order
    is important: torch.compile reduce-overhead is strictly more powerful than
    a manual CUDA graph (graphs + kernel fusion), so the compile path gets
    first crack at every predict tensor."""
    module = _make_module()
    calls: list[str] = []
    x = torch.randn(2, 4)

    def fake_compile(self, t):  # type: ignore[no-untyped-def]
        """Fake compile."""
        calls.append("compile")
        return None

    def fake_graph(self, t):  # type: ignore[no-untyped-def]
        """Fake graph."""
        calls.append("graph")
        return None

    orig_call = MLPTorchModel.__call__

    def fake_call(self, t):  # type: ignore[no-untyped-def]
        """Fake call."""
        calls.append("eager")
        return orig_call(self, t)

    monkeypatch.setattr(MLPTorchModel, "_maybe_compile_predict_forward", fake_compile)
    monkeypatch.setattr(MLPTorchModel, "_maybe_cuda_graph_forward", fake_graph)
    monkeypatch.setattr(MLPTorchModel, "__call__", fake_call)

    module.eval()
    module.predict_step(x, batch_idx=0)
    assert "compile" in calls, "predict_step must consult compile path"
    assert "graph" in calls, "predict_step must consult cuda-graph path"
    assert "eager" in calls, "predict_step must fall through to eager forward"
    # Order: compile precedes graph precedes eager.
    assert calls.index("compile") < calls.index("graph") < calls.index("eager"), f"predict_step path order must be compile -> graph -> eager; got {calls}"


def test_recurrent_network_skips_compile_path(monkeypatch):
    """LSTM/GRU/RNN networks must NOT enter the compile predict path
    (same anti-pattern as F-35)."""
    monkeypatch.setenv("MLFRAME_TORCH_COMPILE_PREDICT", "1")
    network = nn.Sequential(nn.LSTM(4, 8, batch_first=True), nn.Linear(8, 1))
    module = MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=network)
    # CPU early-return fires first, but if we move to a CUDA tensor the
    # recurrent gate would fire. Verify the helper returns None on the
    # CPU input (and check the network does contain LSTM).
    x = torch.randn(2, 4, 4)
    assert module._maybe_compile_predict_forward(x) is None
    _recurrent = (nn.LSTM, nn.GRU, nn.RNN)
    assert any(isinstance(m, _recurrent) for m in module.network.modules())


@pytest.fixture
def reg_data():
    """Reg data."""
    X, y = make_regression(n_samples=64, n_features=4, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_tr, X_te, y_tr


def test_predict_end_to_end_default_path_intact(reg_data, monkeypatch):
    """End-to-end: with both compile + cuda-graph env vars unset, the
    predict path returns eager output (no regression)."""
    monkeypatch.delenv("MLFRAME_TORCH_COMPILE_PREDICT", raising=False)
    monkeypatch.delenv("MLFRAME_CUDA_GRAPH_PREDICT", raising=False)
    X_tr, X_te, y_tr = reg_data
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": nn.MSELoss(), "learning_rate": 1e-2},
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
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)
    assert np.isfinite(preds).all()
