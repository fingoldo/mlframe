"""F-58 (2026-05-31) regression test: the CUDA-graph predict cache
(F-40) returned an UNINITIALISED output buffer on the first call to a
given shape, because the capture-only path skipped the explicit
``g.replay()`` after capture. ``torch.cuda.CUDAGraph()`` records kernels
during the ``with torch.cuda.graph(g):`` block but does NOT execute them
in a way that leaves the output tensor in a usable state -- the buffer
content is undefined (in practice on the tested host: literal zeros).

The bug:
  * The FIRST batch of the FIRST predict() call on a new shape returned
    zeros (or other uninitialised data) instead of the network's
    actual output.
  * Subsequent batches with the same shape used the cached graph and
    were correct -- this masked the bug as a "first-batch random
    failure" only visible in aggregate metrics.
  * Concretely on a tiny 1200-row tabular regression: vanilla PyTorch
    with identical (arch, optim, lr, data, epochs) converged to
    R^2=0.998; mlframe's PytorchLightningRegressor with this bug
    converged to R^2=0.659 -- the entire 0.34 R^2 gap was the
    first-batch zeros.

Fix: insert ``_g.replay()`` after the capture but before returning
``_static_out.clone()``. Adds ~3 us to the first-batch wall-time only;
subsequent batches use the normal cache-hit path which already replays.

Without this guard a SILENT correctness regression rides on every
new-shape predict call -- exactly the failure pattern that wedged the
upstream test ``test_input_normalization_strategies::
test_layernorm_is_ok_on_homogeneous_scale_features`` since F-40
landed (2e52d298's ancestor 2f6fb858).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _make_linear_regression_data(seed: int = 2, n: int = 1200, d: int = 4):
    """Trivial linear regression: y = X @ coefs + small noise.
    OLS reaches R^2 ~= 0.998; a 2-layer ReLU MLP must reach the same
    or the predict path is corrupting output.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    coefs = np.array([1.0, -0.7, 0.5, -0.3], dtype=np.float32)
    y = (X @ coefs + 0.05 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _build_estimator(max_epochs: int = 50) -> PytorchLightningRegressor:
    return PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={
            "loss_fn": torch.nn.MSELoss(),
            "learning_rate": 5e-3,
        },
        network_params={
            "nlayers": 2,
            "first_layer_num_neurons": 32,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 64, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": max_epochs,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "auto",  # Lightning will pick CUDA if available
            "logger": False,
        },
    )


def test_f58_first_batch_predict_matches_manual_forward():
    """The strictest regression check: predict() output must equal a
    manual ``network(X)`` forward to within float32 epsilon. Pre-fix
    they diverged by ~3.9 on the first 64 samples (zeros vs real values).
    """
    X, y = _make_linear_regression_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=2)

    torch.manual_seed(0)
    np.random.seed(0)
    reg = _build_estimator(max_epochs=10)  # short fit; convergence not needed for this check
    reg.fit(X_tr, y_tr)
    pred_predict = reg.predict(X_te)

    reg.model.eval()
    reg.network.eval()
    with torch.no_grad():
        # Move the network back to CPU for a clean reference forward;
        # otherwise device mismatch surfaces.
        net_cpu = reg.network.cpu()
        pred_manual = net_cpu(torch.from_numpy(X_te)).squeeze(-1).numpy()

    max_diff = float(np.abs(pred_predict - pred_manual).max())
    assert max_diff < 1e-3, (
        f"F-58 regression: reg.predict(X) and manual network(X) diverged "
        f"by {max_diff:.4f}. Pre-fix the first-batch CUDA-graph buffer was "
        f"uninitialised. If the diff is concentrated in the first 64 "
        f"samples (the first capture's batch size), the CUDA-graph "
        f"capture path is missing the post-capture _g.replay() call."
    )


def test_f58_regression_converges_to_ols():
    """Honest R^2 contract: a 2-layer ReLU MLP on a trivial linear
    regression with 50 epochs MUST reach R^2 > 0.95 -- vanilla PyTorch
    with the same hyperparameters reaches 0.998. Pre-fix this dropped
    to ~0.66 because the FIRST predict batch returned zeros and the
    other batches were diluted in aggregate metrics.

    Threshold at 0.95 instead of 0.99 to leave a safety margin for
    seed-to-seed variance on different hosts without re-introducing
    the F-58 silent-correctness window.
    """
    X, y = _make_linear_regression_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=2)

    torch.manual_seed(0)
    np.random.seed(0)
    reg = _build_estimator(max_epochs=50)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    r2 = r2_score(y_te, pred)
    assert r2 > 0.95, (
        f"F-58 regression: trivial 4D linear-target MLP only reached "
        f"R^2={r2:.4f}; the underlying network achieves ~0.998 (see the "
        f"sibling test_f58_first_batch_predict_matches_manual_forward). "
        f"If this fires, the F-40 CUDA-graph predict cache is again "
        f"returning uninitialised buffers on the first call to a new "
        f"shape -- check that _g.replay() runs AFTER capture."
    )


def test_f58_disabled_via_env_still_correct(monkeypatch):
    """Sanity gate: disabling F-40 entirely via the env var
    MLFRAME_CUDA_GRAPH_PREDICT=0 must also reach R^2 > 0.95. This pins
    the eager fallback path so any future regression that hits ONLY
    one of the two paths is caught.
    """
    monkeypatch.setenv("MLFRAME_CUDA_GRAPH_PREDICT", "0")
    X, y = _make_linear_regression_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=2)

    torch.manual_seed(0)
    np.random.seed(0)
    reg = _build_estimator(max_epochs=50)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    r2 = r2_score(y_te, pred)
    assert r2 > 0.95, (
        f"F-58 sanity: eager-fallback (MLFRAME_CUDA_GRAPH_PREDICT=0) "
        f"reached only R^2={r2:.4f}; expected > 0.95 on this trivial "
        f"4D linear-target MLP. If this fires, the eager path itself "
        f"is broken (independent of F-40 CUDA-graph)."
    )
