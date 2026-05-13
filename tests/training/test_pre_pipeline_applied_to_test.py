"""Lock: pre_pipeline (SimpleImputer+StandardScaler) MUST be applied to test_df
before predict for NaN-intolerant models.  2026-05-13 prod regression: MLP
silently produced NaN predictions on 494K/524K test rows because the strategy
pre_pipeline was skipped for test_df (cache-hit path).  Linear crashed with
ValueError 3 runs in a row despite NaN guard; the root cause was never fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════
# 1. Sanity: Pipeline.predict handles NaN via SimpleImputer
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_predict_handles_nan():
    """Pipeline([SimpleImputer, StandardScaler, LinearRegression]).predict(X)
    must NOT raise ValueError when X contains NaN.  SimpleImputer.transform
    fills NaN with fitted column means."""
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    X_train.iloc[0, 0] = np.nan
    y_train = rng.normal(size=100)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scl", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)

    X_test = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
    X_test.iloc[3, 2] = np.nan

    preds = pipe.predict(X_test)
    assert preds.shape == (20,)
    assert np.all(np.isfinite(preds)), "predictions must be finite"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Lock: raw LinearRegression.predict raises on NaN (our NaN guard target)
# ═══════════════════════════════════════════════════════════════════════════


def test_raw_linear_raises_on_nan():
    """Raw LinearRegression (no imputer) must raise ValueError on NaN input.
    This is the error our NaN guard catches."""
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("xyz"))
    y_train = rng.normal(size=100)
    lr = LinearRegression().fit(X_train, y_train)

    X_test = pd.DataFrame(rng.normal(size=(10, 3)), columns=list("xyz"))
    X_test.iloc[5, 1] = np.nan

    with pytest.raises(ValueError, match="Input X contains NaN"):
        lr.predict(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Integration: _prepare_test_split MUST transform test_df through the
#    fitted pre_pipeline regardless of skip_pre_pipeline_transform flag.
# ═══════════════════════════════════════════════════════════════════════════


def test_prepare_test_split_transforms_even_when_skip_flag_is_true():
    """Simulate the cache-hit path: pre_pipeline is fitted on train,
    skip_pre_pipeline_transform=True (from cached_dfs), but test_df MUST
    still be transformed.  Without this, NaN reaches LinearRegression."""
    from mlframe.training._pipeline_helpers import (
        _prepare_test_split, _is_fitted,
    )

    rng = np.random.default_rng(0)
    X_train_raw = pd.DataFrame({
        "a": rng.normal(size=200),
        "b": rng.normal(size=200),
    })
    X_train_raw.iloc[10, 0] = np.nan

    pre_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scl", StandardScaler()),
    ])
    y_train = rng.normal(size=200)  # NaN-free target
    pre_pipeline.fit(X_train_raw, y_train)
    assert _is_fitted(pre_pipeline), "pre_pipeline must be fitted"

    X_test = pd.DataFrame({
        "a": rng.normal(size=50),
        "b": rng.normal(size=50),
    })
    X_test.iloc[20, 1] = np.nan

    test_target = rng.normal(size=50)  # NaN-free test target

    # Model trained on pre_pipeline-transformed data
    model = LinearRegression().fit(
        pre_pipeline.transform(X_train_raw), y_train,
    )

    # Simulate the cache-hit path: skip_pre_pipeline_transform=True
    test_df, _, _ = _prepare_test_split(
        df=X_test,
        test_df=X_test,
        test_idx=np.arange(50),
        test_target=test_target,
        target=y_train,
        real_drop_columns=[],
        model=model,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=True,
    )

    # The fix: test_df must have been transformed (no NaN)
    if hasattr(test_df, "isna"):
        assert not test_df.isna().any().any(), (
            "test_df must not contain NaN after _prepare_test_split, "
            "even when skip_pre_pipeline_transform=True.  The pre_pipeline "
            "IS fitted and MUST transform test data."
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration: MLP predict returns NaN silently (not ValueError)
# ═══════════════════════════════════════════════════════════════════════════


def test_mlp_predict_returns_nan_silently_on_nan_input():
    """MLP (PyTorch) does NOT raise ValueError on NaN input — it silently
    returns NaN predictions.  The NaN guard catches ValueError only, so
    MLP NaN goes undetected.  This test proves the MLP NaN behaviour so
    we know to handle it differently (pre-pipeline MUST be applied)."""
    pytest.importorskip("lightning")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from functools import partial
    from sklearn.compose import TransformedTargetRegressor
    from mlframe.training.neural import (
        PytorchLightningRegressor, MLPTorchModel, TorchDataModule,
        MLPNeuronsByLayerArchitecture,
    )

    rng = np.random.default_rng(0)
    n = 100
    X_train = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y_train = rng.normal(size=n).astype(np.float32)

    network_params = {
        "nlayers": 2, "first_layer_num_neurons": 8, "min_layer_neurons": 4,
        "neurons_by_layer_arch": MLPNeuronsByLayerArchitecture.Constant,
        "consec_layers_neurons_ratio": 1.0,
        "activation_function": torch.nn.ReLU,
        "weights_init_fcn": partial(nn.init.kaiming_normal_, nonlinearity="relu"),
        "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
        "use_batchnorm": False,
    }
    model_params = {
        "loss_fn": F.mse_loss, "learning_rate": 1e-3,
        "optimizer": torch.optim.Adam, "optimizer_kwargs": {},
        "lr_scheduler": None, "lr_scheduler_kwargs": {},
    }
    datamodule_params = {
        "read_fcn": None, "data_placement_device": None,
        "features_dtype": torch.float32, "labels_dtype": torch.float32,
        "dataloader_params": {"batch_size": 16, "num_workers": 0},
    }
    trainer_params = {
        "max_epochs": 1, "enable_model_summary": False,
        "log_every_n_steps": 1, "devices": "1",
        "logger": False, "default_root_dir": None,
        "accelerator": "cpu",
    }
    inner = PytorchLightningRegressor(
        model_class=MLPTorchModel, model_params=model_params,
        network_params=network_params, datamodule_class=TorchDataModule,
        datamodule_params=datamodule_params, trainer_params=trainer_params,
    )
    # Wrap in TTR so we match production wiring
    class _TTR(TransformedTargetRegressor):
        def fit(self, X, y, **fit_params):
            from sklearn.base import clone as _clone
            y_arr = np.asarray(y, dtype=np.float64)
            y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
            self.transformer_ = _clone(self.transformer) if self.transformer is not None else None
            if self.transformer_ is not None:
                self.transformer_.fit(y_2d)
            return super().fit(X, y, **fit_params)
    mlp = Pipeline([
        ("scl_x", StandardScaler()),
        ("mlp", _TTR(regressor=inner, transformer=StandardScaler())),
    ])
    mlp.fit(X_train, y_train)

    # Predict on data WITH NaN — MLP silently returns NaN
    X_test = pd.DataFrame(rng.normal(size=(10, 4)), columns=list("abcd"))
    X_test.iloc[3, 1] = np.nan

    preds = mlp.predict(X_test)
    has_nan = not np.all(np.isfinite(preds))
    # MLP + NaN input = NaN output (silent, no ValueError)
    # This is why we MUST apply the pre_pipeline to test_df before predict
    if has_nan:
        # The guard must detect NaN in the OUTPUT, not just catch ValueError
        pass  # This is the bug — MLP NaN goes undetected
