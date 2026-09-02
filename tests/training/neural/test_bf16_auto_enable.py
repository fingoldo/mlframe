"""F-27: bf16-mixed precision auto-enables on Ampere+ GPUs when caller
does NOT set precision explicitly.

Real CUDA HW is not assumed by the test box (CI may be CPU-only).
The test mocks torch.cuda capability detection to exercise both branches
of the dispatcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _params(precision=None, accelerator="cpu"):
    """Builds MLPTorchModel constructor params with an optional explicit Lightning trainer precision override.

    ``accelerator`` defaults to CPU (so ``safe_accelerator`` returns "cpu"); the bf16 gating test overrides it
    to exercise the cuda branch with the CUDA probes monkeypatched.
    """
    trainer_params = {
        "max_epochs": 1,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "log_every_n_steps": 1,
        "devices": 1,
        "accelerator": accelerator,  # CPU by default so safe_accelerator returns "cpu"
        "logger": False,
    }
    if precision is not None:
        trainer_params["precision"] = precision
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
        "network_params": {
            "nlayers": 1,
            "first_layer_num_neurons": 8,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": trainer_params,
        "random_state": 0,
    }


@pytest.fixture
def reg_data():
    """Small deterministic regression dataset shared by the bf16-auto-enable tests."""
    X, y = make_regression(n_samples=64, n_features=4, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_tr, y_tr


def test_bf16_not_enabled_on_cpu_accelerator(reg_data, monkeypatch):
    """When accelerator resolves to CPU, bf16 must NOT be auto-set
    (bf16 on CPU is slow / unsupported by Lightning's default plugin).

    Asserted on the trainer params the dispatcher actually built. The previous body was a fit() call and a
    comment reasoning that "if we got here without crashing, the precision plumbing worked" -- which is false:
    Lightning ACCEPTS bf16-mixed on CPU (it emits a performance warning, not an error), so a regression that
    drops the cuda/gpu guard and sets the precision unconditionally completes the fit and passes. The negative
    contract has to be read off the params, not inferred from the absence of a crash.
    """
    X_tr, y_tr = reg_data

    captured: dict = {}
    # Patch the alias the fit path actually calls (`import lightning as L` -> `L.Trainer(...)`), not
    # `lightning.pytorch.Trainer`: patching a name the production module does not use captures nothing, which
    # the "Trainer was never constructed" assertion below would surface rather than silently passing.
    import mlframe.training.neural.base._base_fit as _fit_mod

    _orig_lightning = _fit_mod.L

    class _CapturingLightning:
        """Delegates everything to the real `lightning` module, recording the Trainer kwargs on the way past."""

        def __getattr__(self, name):
            """Forward every attribute except Trainer."""
            return getattr(_orig_lightning, name)

        @staticmethod
        def Trainer(*args, **kwargs):
            """Record the params the estimator hands Lightning, then build the real Trainer."""
            captured.update(kwargs)
            return _orig_lightning.Trainer(*args, **kwargs)

    monkeypatch.setattr(_fit_mod, "L", _CapturingLightning())

    reg = PytorchLightningRegressor(**_params())
    reg.fit(X_tr, y_tr)

    assert captured, "the Trainer was never constructed; this test needs updating"
    _resolved = str(captured.get("accelerator", "")).lower()
    assert _resolved not in ("cuda", "gpu"), f"this box resolved to {_resolved!r}; the CPU contract is untested here"
    assert "bf16" not in str(captured.get("precision", "")), f"bf16 was auto-enabled on a CPU accelerator: precision={captured.get('precision')!r}"


def test_caller_precision_setting_is_not_overridden(reg_data):
    """Explicit precision='32-true' must NOT be overridden by the
    Ampere+ auto-default."""
    X_tr, y_tr = reg_data
    params = _params(precision="32-true")
    reg = PytorchLightningRegressor(**params)
    reg.fit(X_tr, y_tr)
    # The trainer_params dict on the estimator should still hold the
    # user-supplied precision.
    assert reg.trainer_params["precision"] == "32-true"


def _capture_trainer_params(monkeypatch, reg_data, *, compute_capability):
    """Run the REAL fit dispatcher with CUDA reporting `compute_capability`, and return the params it built.

    `L.Trainer` is replaced by a recorder that raises a sentinel, so the production gating block runs in full
    while no GPU fit is ever started.
    """
    import mlframe.training.neural.base._base_fit as _fit_mod

    captured: dict = {}

    class _Sentinel(RuntimeError):
        """Aborts the fit immediately after the params are built."""

    _orig_lightning = _fit_mod.L

    class _CapturingLightning:
        """Delegates to the real `lightning`, recording Trainer kwargs and stopping the fit there."""

        def __getattr__(self, name):
            """Forward every attribute except Trainer."""
            return getattr(_orig_lightning, name)

        @staticmethod
        def Trainer(*args, **kwargs):
            """Record and abort."""
            captured.update(kwargs)
            raise _Sentinel

    monkeypatch.setattr(_fit_mod, "L", _CapturingLightning())
    monkeypatch.setattr(_fit_mod, "safe_accelerator", lambda requested: "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: compute_capability)

    X_tr, y_tr = reg_data
    reg = PytorchLightningRegressor(**_params(accelerator="cuda"))
    try:
        reg.fit(X_tr, y_tr)
    except _Sentinel:
        pass
    assert captured, "the dispatcher never reached Trainer construction; this test needs updating"
    return captured


def test_bf16_auto_enabled_on_ampere_and_not_before_it(reg_data, monkeypatch):
    """The dispatcher's own gating, exercised through PRODUCTION code.

    This test used to transcribe the gating rule into its own body -- building a local `trainer_params` dict,
    running its own copy of the `if "precision" not in ... and _resolved in ("cuda","gpu")` block, and asserting
    on that -- so it verified the test's transcription and never executed the dispatcher at all. Its only
    production import was a `# noqa: F401` name it never called. A regression in the real gating (a dropped
    compute-capability check, an inverted comparison, a moved branch) could not fail it.
    """
    ampere = _capture_trainer_params(monkeypatch, reg_data, compute_capability=(8, 0))
    assert ampere.get("precision") == "bf16-mixed", f"Ampere+ (cc=8.0) should auto-enable bf16-mixed; got {ampere.get('precision')!r}"

    pre_ampere = _capture_trainer_params(monkeypatch, reg_data, compute_capability=(7, 5))
    assert "bf16" not in str(pre_ampere.get("precision", "")), f"pre-Ampere (cc=7.5) should NOT auto-enable bf16-mixed; got {pre_ampere.get('precision')!r}"
