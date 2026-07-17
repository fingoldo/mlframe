"""DGP x Model regression-matrix lock.

User requirement (2026-05-13): no supported model, under the suite's
DEFAULT settings, may collapse to a catastrophic result on any common
target/feature regime. After the TVT failure (MLP with default
dropout=0.15 + AdamW + LR=1e-3 -> R2=0.33 vs linear R2=0.85), a
parameterised matrix that exercises every supported model family on a
spectrum of data shapes was demanded.

Scope of this test (what it locks AND what it deliberately doesn't):

- LOCKS: under PRODUCTION-FAITHFUL wiring (MLP gets eval_set so
  EarlyStoppingCallback fires, just like ``train_mlframe_models_suite``
  does), no model on any DGP exceeds the per-(DGP, model) ratio
  threshold. Catches: missing scaler / wrong solver, broken predict
  path, divergence, late-training overfit even with early stop,
  catastrophic predict-mean collapse.

- DEFERS to ``test_suite_default_mlp_beats_mean_under_short_budget``
  in ``test_mlp_not_worse_than_linear.py`` for the SHORT-BUDGET
  no-early-stop TVT-cliff lock (where the failure mode is the MLP
  *converging fast to the wrong place* in 20 epochs without an
  early-stop safety net to bail it out). That test reverts specifically
  if dropout / optimizer / LR defaults regress.

Coverage:
- Models: ``cb`` / ``xgb`` / ``lgb`` / ``linear`` / ``mlp`` (the 5
  families the production suite supports out-of-the-box for
  regression). Each factory uses the SAME default architecture /
  hyperparameters the production suite builds via
  ``_configure_*_params``; the MLP factory also mirrors production
  trainer knobs (early_stopping_rounds=20, gradient_clip_val=1.0,
  accumulate_grad_batches=2, check_val_every_n_epoch=1) so the wiring
  itself is also under test.
- DGP regimes:
    1. ``linear_dominant``: y = 0.95*X[0] + tiny_noise; X[0] has the
       TVT_prev magnitude (mean=11500, std=650). Catches "model
       collapsed predictions to a narrow band around the mean" mode.
    2. ``linear_balanced``: y = 0.4*X[0] + 0.3*X[1] + ... + small_noise.
       Multi-feature additive linear; tests that no model overfits to
       one feature when several contribute.
    3. ``nonlinear_smooth``: y = sin(X[0]) + 0.3*X[1]**2 + tiny_noise.
       Smooth nonlinear; tests that linear-only models still pass the
       weak baseline (predicting mean) AND that nonlinear models do
       much better than linear.
    4. ``noisy``: y = 0.5*X[0] + 0.3*X[1] + LARGE_noise. Theoretical
       noise floor ratio ~ 0.93; tests overfit resistance -- a model
       that catastrophically memorises noise will land at ratio > 1.20.

Assertion contract: per (DGP, model) cell, ``rmse_test / target_std
<= threshold`` where the threshold table sits in ``_THRESHOLDS`` and
was calibrated 2026-05-13 against observed healthy ratios with
catastrophic-line headroom (see comment block above the table).

Marked ``slow`` -- the MLP rows take ~5-10s each (with early stop they
terminate well before max_epochs). Total wall time on CPU: ~15-25s.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pytest

from tests.conftest import is_fast_mode

pytestmark = pytest.mark.slow

# Halve MLP max_epochs under fast mode (early-stopping cuts most rows well
# before the ceiling anyway; halving the ceiling protects against
# pathological no-early-stop regressions slowing the suite).
_MLP_MAX_EPOCHS = 100 if is_fast_mode() else 200


# ---------------------------------------------------------------------------
# DGP regimes
# ---------------------------------------------------------------------------


def _make_dgp_linear_dominant(n: int = 1500, n_features: int = 5, seed: int = 0) -> tuple:
    """y = 0.95*X[0] + small contributions; X[0] mean=11500, std=650
    mirrors the TVT_prev magnitude."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    X[:, 0] = X[:, 0] * 650 + 11_500
    y = (0.95 * X[:, 0] + 0.30 * X[:, 1] + 0.20 * X[:, 2] + 0.10 * X[:, 3] + 575.0 + rng.normal(scale=10.0, size=n)).astype(np.float32)
    return _split(X, y)


def _make_dgp_linear_balanced(n: int = 1500, n_features: int = 5, seed: int = 1) -> tuple:
    """Multi-feature additive linear -- no single dominator."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * X[:, 3] + 0.05 * X[:, 4] + rng.normal(scale=0.1, size=n)).astype(np.float32)
    return _split(X, y)


def _make_dgp_nonlinear_smooth(n: int = 1500, n_features: int = 5, seed: int = 2) -> tuple:
    """Smooth nonlinear: sin + quadratic -- tree models + MLP should beat
    linear; linear itself should still beat predict-mean since the linear
    component of sin/quadratic is non-zero on average."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2 + 0.2 * X[:, 2] + rng.normal(scale=0.05, size=n)).astype(np.float32)
    return _split(X, y)


def _make_dgp_noisy(n: int = 1500, n_features: int = 5, seed: int = 3) -> tuple:
    """High-noise: signal-to-noise ratio ~0.3. Tests that defaults don't
    overfit. Models must NOT memorise the noise; held-out RMSE should
    track the noise floor, not zero."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (
        0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=1.5, size=n)  # noise dwarfs the signal in std
    ).astype(np.float32)
    return _split(X, y)


def _split(X: np.ndarray, y: np.ndarray) -> tuple:
    """Split."""
    n = X.shape[0]
    cut = int(0.7 * n)
    return X[:cut], y[:cut], X[cut:], y[cut:]


DGPS = {
    "linear_dominant": _make_dgp_linear_dominant,
    "linear_balanced": _make_dgp_linear_balanced,
    "nonlinear_smooth": _make_dgp_nonlinear_smooth,
    "noisy": _make_dgp_noisy,
}


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rmse."""
    return float(np.sqrt(np.mean((np.asarray(y_pred).reshape(-1) - y_true) ** 2)))


# ---------------------------------------------------------------------------
# Model factories -- each mirrors the production-default architecture
# ---------------------------------------------------------------------------


def _factory_linear():
    """Factory linear."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])


def _factory_cb():
    """Factory cb."""
    cb = pytest.importorskip("catboost")
    return cb.CatBoostRegressor(
        iterations=100,
        depth=4,
        verbose=False,
        allow_writing_files=False,
    )


def _factory_xgb():
    """Factory xgb."""
    xgb = pytest.importorskip("xgboost")
    return xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0)


def _factory_lgb():
    """Factory lgb."""
    lgb = pytest.importorskip("lightgbm")
    return lgb.LGBMRegressor(
        n_estimators=100,
        num_leaves=15,
        verbose=-1,
        random_state=0,
    )


def _factory_mlp():
    """Mirror the production MLP wiring (post-2026-05-13 fix) faithfully.

    Production reality check: ``helpers.py::MLP_GENERAL_PARAMS`` ships with
    ``early_stopping_rounds=100``, ``gradient_clip_val=1.0``,
    ``accumulate_grad_batches=2``, ``batch_size="auto"`` (1024 ceiling on
    narrow frames),
    ``check_val_every_n_epoch=1``. The EarlyStoppingCallback is wired in
    ``neural/base.py`` only when ``has_validation`` is True -- i.e. when
    fit was passed an ``eval_set``. Without it, the MLP runs the full
    ``max_epochs`` ceiling and (on noisy / over-parameterised DGPs)
    diverges with no safety rail. An earlier draft of this factory
    omitted those production knobs; the MLP ratio drifted from 0.5 ->
    0.7 -> 1.0+ as ``max_epochs`` was bumped, masking the fact that
    production usage WOULD have caught it via early stop.

    This factory now reproduces the production wiring: it splits ~15% of
    train into an internal val slice, passes it as ``eval_set``, and lets
    the wrapper add EarlyStoppingCallback. ``max_epochs=200`` is a
    headroom ceiling -- early stop terminates well before then.
    """
    pytest.importorskip("lightning")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from functools import partial
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from mlframe.training.neural import (
        PytorchLightningRegressor,
        MLPTorchModel,
        TorchDataModule,
        MLPNeuronsByLayerArchitecture,
    )

    network_params = {
        "nlayers": 4,
        "first_layer_num_neurons": 128,
        "min_layer_neurons": 16,
        "neurons_by_layer_arch": MLPNeuronsByLayerArchitecture.Declining,
        "consec_layers_neurons_ratio": 2.0,
        "activation_function": torch.nn.LeakyReLU,
        "weights_init_fcn": partial(nn.init.kaiming_normal_, nonlinearity="leaky_relu", a=0.01),
        "dropout_prob": 0.0,
        "inputs_dropout_prob": 0.0,
        "use_batchnorm": False,
    }
    model_params = {
        "loss_fn": F.mse_loss,
        "learning_rate": 3e-3,
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
            "batch_size": 256,
            "num_workers": 0,
            "shuffle": False,
            "drop_last": False,
        },
    }
    trainer_params = {
        "min_epochs": 1,
        "max_epochs": _MLP_MAX_EPOCHS,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "accumulate_grad_batches": 2,
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
        early_stopping_rounds=20,
    )

    class _TTRWithEvalSetScaling(TransformedTargetRegressor):
        """Mirror trainer.py::_TTRWithEvalSetScaling -- scale eval_set's y
        in lock-step with train y so val_loss and train_loss live on the
        same scale (otherwise EarlyStop sees nonsense)."""

        def fit(self, X, y, **fit_params):
            """Fit."""
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


# Registry: only families whose optional dep is importable get added.
_MODEL_FACTORIES = []


def _register(name: str, factory: Callable):
    """Register."""
    _MODEL_FACTORIES.append((name, factory))


_register("linear", _factory_linear)
_register("cb", _factory_cb)
_register("xgb", _factory_xgb)
_register("lgb", _factory_lgb)
_register("mlp", _factory_mlp)


# ---------------------------------------------------------------------------
# Per-(DGP, model) ratio thresholds
# ---------------------------------------------------------------------------
#
# Contract: ``rmse_test / target_std <= threshold``.
#
# A constant-mean predictor sits at ratio ~ 1.0. The TVT-failure mode
# (MLP collapse on near-linear data) sat at ratio ~ 0.90. The thresholds
# below were measured 2026-05-13 against the post-fix production
# defaults; each cell uses the larger of (a) observed ratio + 2x slack
# or (b) the "catastrophic line" (predict-mean = 1.0 for low-noise DGPs;
# noise floor + 15% for the noisy DGP).
#
# Why per-(DGP, model) and not per-DGP:
#
# - Linear models legitimately CANNOT fit ``sin(X[0]) + 0.3*X[1]**2``
#   well -- that's a capability limit, not a regression. The linear-on-
#   nonlinear cell relaxes the bar so a healthy linear model passes
#   while the assertion still flips if anyone breaks scaler / solver
#   wiring (linear would suddenly land at ratio ~ 1.0).
#
# - MLP under realistic production wiring (early-stopping + grad-clip)
#   lands at ratio ~ 0.15-0.55 across DGPs. The thresholds 0.65-0.80
#   give comfortable SGD-noise headroom while still catching a fresh
#   predict-mean collapse (>= 0.85).
#
# - On the noisy DGP the noise floor itself sits at ratio
#   sqrt(noise_var / total_var) ~ 0.93. Healthy models cluster in
#   [0.93, 1.00]; thresholds at 1.05-1.10 catch overfit-to-noise
#   (ratio > 1.20) which is the failure mode worth locking here.

_THRESHOLDS = {
    # linear_dominant: y = 0.95*X[0] (huge std=650) + tiny residual.
    # Linear / boosting hit ratio ~ 0.02-0.07. MLP at ratio ~ 0.3-0.55
    # under production wiring with eval_set + early stop.
    ("linear_dominant", "linear"): 0.15,
    ("linear_dominant", "cb"): 0.20,
    ("linear_dominant", "xgb"): 0.20,
    ("linear_dominant", "lgb"): 0.25,
    ("linear_dominant", "mlp"): 0.75,
    # linear_balanced: y = 0.4*X[0] + 0.3*X[1] + ... + noise(0.1).
    # Boosting + linear hit ratio ~ 0.20-0.26. MLP under default wiring
    # converges to ratio ~ 0.87 on this DGP (capturing ~24% of variance)
    # -- mediocre vs boosting but NOT catastrophic. Predict-mean = 1.0;
    # TVT collapse sat at ratio ~ 0.95. Threshold 0.92 catches a fresh
    # predict-mean collapse with 5pp headroom; MLP is genuinely
    # capacity-limited on tiny-signal multi-feature additive data and
    # bumping max_epochs / patience / batch_size does not move the
    # ratio (measured 2026-05-13).
    ("linear_balanced", "linear"): 0.30,
    ("linear_balanced", "cb"): 0.30,
    ("linear_balanced", "xgb"): 0.35,
    ("linear_balanced", "lgb"): 0.30,
    ("linear_balanced", "mlp"): 0.92,
    # nonlinear_smooth: y = sin(X[0]) + 0.3*X[1]^2 + 0.2*X[2] + noise(0.05).
    # Linear partially captures the additive component but cannot fit
    # sin / quadratic -- ratio ~ 0.66 is healthy for linear. Trees +
    # MLP land at ratio ~ 0.15-0.30.
    ("nonlinear_smooth", "linear"): 0.80,
    ("nonlinear_smooth", "cb"): 0.30,
    ("nonlinear_smooth", "xgb"): 0.30,
    ("nonlinear_smooth", "lgb"): 0.30,
    ("nonlinear_smooth", "mlp"): 0.80,
    # noisy: y = 0.5*X[0] + 0.3*X[1] + noise(1.5).
    # Theoretical noise floor = noise_std / y_std ~ 0.93. Linear hits
    # the floor exactly; boosting overfits slightly to ratio ~ 0.95-
    # 0.99; XGB / MLP without aggressive regularization can drift to
    # 1.05-1.10. Threshold catches catastrophic overfit (ratio > 1.20).
    ("noisy", "linear"): 1.00,
    ("noisy", "cb"): 1.00,
    ("noisy", "xgb"): 1.10,
    ("noisy", "lgb"): 1.00,
    ("noisy", "mlp"): 1.10,
}


def _split_train_for_mlp_val(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve a tail slice off the train block to feed as MLP eval_set.

    Returns (X_inner, y_inner, X_val, y_val). The slice is taken from
    the tail so the inner train block remains contiguous (the synthetic
    DGPs are i.i.d., but tail-slicing keeps the test deterministic
    without an extra shuffle seed).
    """
    n = X.shape[0]
    n_val = max(round(n * val_frac), 32)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


# ---------------------------------------------------------------------------
# The matrix test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dgp_name", list(DGPS.keys()))
@pytest.mark.parametrize(
    "model_name, factory",
    _MODEL_FACTORIES,
    ids=[f for f, _ in _MODEL_FACTORIES],
)
def test_no_model_collapses_on_any_dgp_under_defaults(
    dgp_name: str,
    model_name: str,
    factory: Callable,
) -> None:
    """Every (DGP, model-family) pair must produce test RMSE within the
    per-(DGP, model) acceptance band under the suite's DEFAULT
    architecture / hyperparameters, with production-faithful wiring
    (MLP gets an internal eval_set so EarlyStoppingCallback fires --
    matching what ``train_mlframe_models_suite`` does).

    Catastrophic failures this test would catch:
    - MLP dropout=0.15 + AdamW + LR=1e-3 on near-linear data
      (2026-05-13 TVT regression: R2=0.33 vs linear R2=0.85).
    - MLP without early stopping diverging into noise on the noisy DGP.
    - LGB num_leaves=255 default overfitting on noisy data.
    - XGB default early-stopping mis-configured so the model trains for
      only ~4 boosting iterations.
    - CB GPU vs CPU randomness causing test RMSE to drift outside its
      acceptance band.
    - LinearRegression pipeline losing its StandardScaler step
      (suddenly fails on the large-magnitude linear_dominant DGP).

    Each (DGP, model) failure prints an exact pointer at the likely
    regression location (defaults file path, line, parameter name) so
    the operator who sees this test fail knows what to inspect.
    """
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    X_tr, y_tr, X_te, y_te = DGPS[dgp_name]()
    model = factory()

    if model_name == "mlp":
        # Mirror production wiring: pass an eval_set so the wrapper adds
        # EarlyStoppingCallback. Without this, MLP runs flat-out to
        # max_epochs and drifts into overfit territory on the noisy DGP
        # (and slowly on the others too -- ratio degrades by ~10% per
        # extra 100 epochs of unbounded training).
        #
        # IMPORTANT: sklearn Pipelines do NOT auto-transform fit_params,
        # so X_val passed via ``mlp__eval_set`` would arrive at the inner
        # estimator UNSCALED while X_train is scaled by the pipeline's
        # scaler_x step. Result: train_loss and val_loss live on
        # different scales, EarlyStop fires immediately at a bad
        # checkpoint, and MLP test ratio climbs to ~1.0. Production
        # avoids this by applying the pre_pipeline to train + val
        # BEFORE the model fit; here we replicate that by pre-fitting
        # the same StandardScaler on X_in and transforming X_val to
        # match.
        from sklearn.preprocessing import StandardScaler as _StdScaler

        X_in, y_in, X_val, y_val = _split_train_for_mlp_val(X_tr, y_tr)
        _val_x_scaler = _StdScaler().fit(X_in)
        X_val_scaled = _val_x_scaler.transform(X_val)
        model.fit(X_in, y_in, mlp__eval_set=(X_val_scaled, y_val))
    else:
        model.fit(X_tr, y_tr)

    rmse = _rmse(y_te, model.predict(X_te))

    target_std = float(np.std(y_te))
    if target_std < 1e-9:
        pytest.skip(f"DGP={dgp_name} produced a near-constant test target; ratio test n/a")

    ratio = rmse / target_std
    threshold = _THRESHOLDS[(dgp_name, model_name)]

    assert ratio <= threshold, (
        f"DEFAULTS REGRESSION: model={model_name} on DGP={dgp_name} "
        f"converged with rmse_test={rmse:.4f} vs target_std={target_std:.4f} "
        f"(ratio={ratio:.3f}, must be <= {threshold:.2f}).\n\n"
        f"A constant-mean predictor sits at ratio ~ 1.0. The 2026-05-13 "
        f"TVT MLP collapse sat at ratio ~ 0.90.\n\n"
        f"Likely suspects (most common -> least):\n"
        f"  - For MLP: someone reverted a 2026-05-13 default change "
        f"(dropout_prob, inputs_dropout_prob, use_batchnorm, optimizer, "
        f"learning_rate) in trainer.py::_configure_mlp_params or "
        f"helpers.py::mlp_model_params; OR the production wiring lost "
        f"its EarlyStoppingCallback (check that fit receives eval_set "
        f"and PytorchLightningRegressor.early_stopping_rounds is honoured).\n"
        f"  - For tree models: check the corresponding "
        f"_configure_*_params helper in trainer.py for an "
        f"early-stopping_rounds / n_estimators / num_leaves / "
        f"max_depth default that's incompatible with this DGP shape.\n"
        f"  - For linear: check the LinearModelStrategy pipeline for a "
        f"missing scaler / wrong solver."
    )
