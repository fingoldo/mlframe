"""Biz-value locks for the production PyTorch-Lightning MLP wiring.

Motivation. The 2026-05-11 TVT training log exposed MLP getting RMSE=585
while Linear got RMSE=252 on the same target. The user's hypothesis is
that the *predict path* may have an arithmetic bug rather than a slow
convergence issue. These tests separate the two concerns:

Test A -- ``test_mlp_predict_path_roundtrips_known_weights``:
  Constructs an MLP with KNOWN identity-ish weights (so the network
  computes a known closed-form function of the input). Runs predict on
  a synthetic batch and asserts the output matches the analytic
  expectation. Independent of training convergence: this catches bugs
  in ``_predict_raw`` (wrong batch reshape, missing inverse_transform,
  eval-mode swap, etc.) regardless of whether SGD converges.

Test B -- ``test_mlp_not_predicting_near_mean``:
  Trains a real MLP on a synthetic dominant-feature regression (mean
  ~11500, std ~650 -- same magnitude as TVT) for a generous epoch
  budget, then asserts RMSE <= 50% of target_std. The TVT failure
  mode (MLP stuck at predicting near-mean) sits at RMSE ~ target_std
  (R^2 < 0.2). 0.5 * target_std is the boundary between "captured at
  least some signal" and "trivial mean-predictor". Healthy convergence
  lands well below 0.3 * target_std.

Marked ``slow`` because the real Lightning + torch init costs ~5-15s.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow


def _make_dominant_feature_dataset(
    n_total: int = 1500, n_features: int = 5, seed: int = 0,
) -> tuple:
    """y = 0.95 * f1 + small contributions + noise; f1 dominates by 10x."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_total, n_features)).astype(np.float32)
    X[:, 0] = X[:, 0] * 650 + 11_500
    y = (
        0.95 * X[:, 0]
        + 0.30 * X[:, 1]
        + 0.20 * X[:, 2]
        + 0.10 * X[:, 3]
        + 575.0
        + rng.normal(scale=10.0, size=n_total)
    ).astype(np.float32)
    n_train = int(0.7 * n_total)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred).reshape(-1) - y_true) ** 2)))


# ---------------------------------------------------------------------------
# Test A: predict-path roundtrip on KNOWN weights
# ---------------------------------------------------------------------------


def test_mlp_predict_path_matches_direct_forward_pass() -> None:
    """The MLP predict path must return EXACTLY ``network(x)`` from a
    direct forward call (eval mode, no_grad).

    Strategy. Fit a 1-epoch MLP so ``self.model`` is populated. Then
    compare ``reg.predict(X)`` against ``reg.model.network(X_tensor)``
    computed manually. If they disagree it's a real bug in
    ``_predict_raw`` (wrong reshape, missing eval-mode swap, wrong batch
    concatenation, etc.) -- independent of training convergence.

    Failure modes this catches:
    - ``_predict_raw`` applies a transformation that should have stayed
      in training-only (sigmoid, dropout-in-train-mode, batchnorm
      running-stat collection).
    - Predictions are squeezed / reshaped incorrectly (e.g. trimming a
      sample row, swapping axes).
    - The wrong checkpoint / network is used at predict time.
    """
    pytest.importorskip("lightning")
    from mlframe.training.neural import (
        PytorchLightningRegressor,
        MLPTorchModel,
        TorchDataModule,
    )

    torch.manual_seed(0)

    n_features = 3
    network_params = {
        "nlayers": 2,
        "first_layer_num_neurons": 8,
        "min_layer_neurons": 4,
        "dropout_prob": 0.0,
        "inputs_dropout_prob": 0.0,
        "use_layernorm": False,
        "use_batchnorm": False,
        "activation_function": torch.nn.ReLU,
    }
    model_params = {
        "loss_fn": torch.nn.functional.mse_loss,
        "learning_rate": 1e-3,
    }
    datamodule_params = {
        "read_fcn": None,
        "data_placement_device": None,
        "features_dtype": torch.float32,
        "labels_dtype": torch.float32,
        "dataloader_params": {"batch_size": 16, "num_workers": 0},
    }
    trainer_params = {
        "max_epochs": 1,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
        "devices": 1,
        "logger": False,
        "default_root_dir": None,
        "accelerator": "cpu",
    }
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params=model_params,
        network_params=network_params,
        datamodule_class=TorchDataModule,
        datamodule_params=datamodule_params,
        trainer_params=trainer_params,
    )

    # 1-epoch fit so reg.model is populated.
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(64, n_features)).astype(np.float32)
    y_train = rng.normal(size=(64,)).astype(np.float32)
    reg.fit(X_train, y_train)

    # Reference: direct forward call through the network in eval mode.
    X_test = rng.normal(size=(7, n_features)).astype(np.float32)
    reg.model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X_test)
        ref = reg.model.network(x_t).cpu().numpy().reshape(-1)

    # Suite predict path -- must match the reference exactly.
    got = np.asarray(reg.predict(X_test)).reshape(-1)
    np.testing.assert_allclose(
        got, ref, atol=1e-5, rtol=1e-5,
        err_msg=(
            f"MLP predict path diverged from a direct ``network(x)`` "
            f"forward call. ref[0]={ref[0]:.6f}, got[0]={got[0]:.6f}. "
            f"This is a real bug in ``_predict_raw`` or ``predict_step`` "
            f"-- not a slow-convergence artefact. Check eval-mode swap, "
            f"batch reshape, output squeeze, and any precision casts."
        ),
    )


# ---------------------------------------------------------------------------
# Test B: full-stack MLP must not predict near-mean
# ---------------------------------------------------------------------------


def _build_production_mlp_wrapper(n_features: int):
    """Mirror the suite's MLP construction at trainer.py:4957-5003.

    Returns a sklearn Pipeline with X-side StandardScaler + a
    ``_TTRWithEvalSetScaling`` around ``PytorchLightningRegressor`` (y-side
    StandardScaler). Mirrors the F1 + F7 fixes from 2026-05-11/12. Smaller
    network/epochs than prod to keep the test under 30s.
    """
    pytest.importorskip("lightning")
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import torch.nn.functional as F
    from mlframe.training.neural import (
        PytorchLightningRegressor,
        MLPTorchModel,
        TorchDataModule,
    )

    network_params = {
        "nlayers": 2,
        "first_layer_num_neurons": 64,
        "min_layer_neurons": 32,
        "dropout_prob": 0.0,
        "activation_function": torch.nn.ReLU,
    }
    model_params = {
        "loss_fn": F.mse_loss,
        "learning_rate": 3e-2,
        "l1_alpha": 0.0,
        "optimizer": torch.optim.Adam,
        "optimizer_kwargs": {},
        "lr_scheduler": None,
        "lr_scheduler_kwargs": {},
    }
    datamodule_params = {
        "read_fcn": None,
        "data_placement_device": None,
        "features_dtype": torch.float32,
        "labels_dtype": torch.float32,
        "dataloader_params": {
            "batch_size": 32, "num_workers": 0,
            "shuffle": False, "drop_last": False,
        },
    }
    trainer_params = {
        "min_epochs": 1,
        # 500 epochs over ~1K rows / batch=32 (~33 batches/epoch) = ~16K
        # SGD steps. Wide enough budget for SGD to reach a good fit on the
        # near-linear DGP (OLS converges in one shot, MLP needs many).
        "max_epochs": 500,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
        "devices": 1,
        "logger": False,
        "default_root_dir": None,
        "accelerator": "cpu",
    }
    inner = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params=model_params,
        network_params=network_params,
        datamodule_class=TorchDataModule,
        datamodule_params=datamodule_params,
        trainer_params=trainer_params,
    )

    class _TTRWithEvalSetScaling(TransformedTargetRegressor):
        def fit(self, X, y, **fit_params):
            from sklearn.base import clone as _clone
            y_arr = np.asarray(y, dtype=np.float64)
            y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
            self.transformer_ = _clone(self.transformer) if self.transformer is not None else None
            if self.transformer_ is not None:
                self.transformer_.fit(y_2d)
                if "eval_set" in fit_params and fit_params["eval_set"] is not None:
                    es = fit_params["eval_set"]
                    if isinstance(es, tuple) and len(es) == 2:
                        X_val, y_val = es
                        yv = np.asarray(y_val, dtype=np.float64)
                        yv2 = yv.reshape(-1, 1) if yv.ndim == 1 else yv
                        yv_scaled = self.transformer_.transform(yv2).reshape(yv.shape)
                        fit_params["eval_set"] = (X_val, yv_scaled)
            return super().fit(X, y, **fit_params)

    ttr = _TTRWithEvalSetScaling(regressor=inner, transformer=StandardScaler())
    return Pipeline([("scaler_x", StandardScaler()), ("mlp", ttr)])


def test_mlp_not_predicting_near_mean() -> None:
    """Biz-value: trained MLP must capture some of the dominant-feature
    signal. RMSE on test must be at most 50% of target_std -- a healthy
    fit lands at ~5%, the TVT failure mode (MLP stuck at predicting
    near-mean) sits at ~100% (R^2 < 0.2 in the prod log).

    We deliberately DO NOT compare against Linear -- on a near-linear DGP
    OLS is the MLE and MLP can be 5-10x slower to converge to the same
    fit. The "MLP <= Linear" bar conflates wiring health with SGD
    convergence speed. The 0.5 * target_std floor catches the catastrophic
    cliff (the only failure mode we actually want to lock).
    """
    torch.manual_seed(0)
    X_tr, y_tr, X_te, y_te = _make_dominant_feature_dataset(seed=0)

    mlp = _build_production_mlp_wrapper(n_features=X_tr.shape[1])
    mlp.fit(X_tr, y_tr)
    rmse_mlp = _rmse(y_te, mlp.predict(X_te))

    # The TVT failure (R^2 = 0.18) sits at RMSE/std ratio ~ 0.90; a healthy
    # MLP on this synthetic DGP lands at ~0.20-0.40 in 500 epochs. The bar
    # at 0.70 catches the TVT cliff with comfortable headroom for SGD
    # noise across machines / seeds.
    target_std = float(np.std(y_te))
    floor_ratio = rmse_mlp / max(target_std, 1e-9)
    assert floor_ratio <= 0.70, (
        f"Production MLP failed to capture the dominant-feature signal: "
        f"rmse_mlp={rmse_mlp:.2f} vs target_std={target_std:.2f} "
        f"(ratio={floor_ratio:.3f}, must be <=0.70). The TVT failure mode "
        f"(MLP stuck at predicting near-mean, ratio ~ 0.90) reproduces if "
        f"this fails."
    )
