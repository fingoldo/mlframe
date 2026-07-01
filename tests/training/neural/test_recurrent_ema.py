"""F-68 (2026-05-31) tests for recurrent EMA. Mirrors MLP's F-28 EMA
wiring for the recurrent path: when use_ema=True, Lightning's
WeightAveraging callback maintains an EMA copy of the weights and
swaps it in on on_train_end so predict() uses the averaged weights.

Two layers:
  1. Config-surface: ``use_ema`` + ``ema_decay`` fields exist with the
     right defaults.
  2. Integration: ``RecurrentRegressorWrapper.fit(...)`` with
     ``use_ema=True`` adds an EMA callback to the Lightning Trainer
     and the fit converges without exceptions.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch

from mlframe.training.neural._recurrent_config import (
    InputMode,
    RecurrentConfig,
    RNNType,
)


# --- Config surface --------------------------------------------------------


def test_recurrent_config_exposes_ema_defaults():
    cfg = RecurrentConfig()
    assert hasattr(cfg, "use_ema")
    assert hasattr(cfg, "ema_decay")
    assert cfg.use_ema is False
    assert cfg.ema_decay == 0.999


def test_recurrent_config_accepts_custom_ema_kwargs():
    cfg = RecurrentConfig(use_ema=True, ema_decay=0.995)
    assert cfg.use_ema is True
    assert cfg.ema_decay == 0.995


# --- Integration -----------------------------------------------------------


def test_recurrent_fit_with_ema_does_not_crash():
    """Smoke: recurrent regression fit + predict with use_ema=True
    completes and the predictions are finite. Without an EMA callback
    fit() doesn't crash either; this test guards against the EMA wiring
    introducing a callback misconfig (e.g. invalid avg_fn signature,
    missing import path)."""
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper

    rng = np.random.default_rng(0)
    n = 80
    seq_len = 8
    n_seq_feats = 3
    n_aux_feats = 2
    sequences = [
        rng.normal(size=(seq_len, n_seq_feats)).astype(np.float32)
        for _ in range(n)
    ]
    features = rng.normal(size=(n, n_aux_feats)).astype(np.float32)
    y = (
        np.array([s.mean() for s in sequences], dtype=np.float32)
        + features[:, 0]
    )

    cfg = RecurrentConfig(
        rnn_type=RNNType.LSTM,
        input_mode=InputMode.HYBRID,
        hidden_size=16,
        num_layers=1,
        max_epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        accelerator="cpu",
        scale_features=True,
        num_workers=0,
        use_ema=True,
        ema_decay=0.99,
        early_stopping_patience=10,
    )
    reg = RecurrentRegressorWrapper(config=cfg)
    reg.fit(features=features, labels=y, sequences=sequences)
    pred = reg.predict(features=features[:20], sequences=sequences[:20])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all(), (
        f"EMA-wrapped recurrent predictions had non-finite values: {pred}"
    )


def test_recurrent_ema_callback_attached_when_use_ema_true(monkeypatch):
    """White-box: when use_ema=True, _create_trainer attaches either
    a WeightAveraging or StochasticWeightAveraging callback to the
    Trainer's callbacks list. We don't care which (depends on Lightning
    version); we DO care that *something* was added."""
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper
    import lightning as L

    cfg = RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        max_epochs=1, batch_size=8,
        accelerator="cpu",
        use_ema=True, ema_decay=0.99,
        early_stopping_patience=1,
    )
    reg = RecurrentRegressorWrapper(config=cfg)
    trainer, _ = reg._create_trainer(has_validation=False, plot=False)

    try:
        from lightning.pytorch.callbacks import WeightAveraging
        _native = True
    except ImportError:
        _native = False
    from lightning.pytorch.callbacks import StochasticWeightAveraging

    expected_type = WeightAveraging if _native else StochasticWeightAveraging  # type: ignore[possibly-unbound]
    assert any(isinstance(cb, expected_type) for cb in trainer.callbacks), (
        f"use_ema=True did not attach {expected_type.__name__} to Trainer.callbacks; "
        f"observed: {[type(cb).__name__ for cb in trainer.callbacks]}"
    )


def test_recurrent_no_ema_callback_when_use_ema_false():
    """Default off: no WeightAveraging / SWA callback attached."""
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper

    cfg = RecurrentConfig(
        input_mode=InputMode.FEATURES_ONLY,
        max_epochs=1, batch_size=8,
        accelerator="cpu",
        use_ema=False,
    )
    reg = RecurrentRegressorWrapper(config=cfg)
    trainer, _ = reg._create_trainer(has_validation=False, plot=False)

    try:
        from lightning.pytorch.callbacks import WeightAveraging
        bad_types: tuple[type, ...] = (WeightAveraging,)
    except ImportError:
        bad_types = ()
    from lightning.pytorch.callbacks import StochasticWeightAveraging
    bad_types = bad_types + (StochasticWeightAveraging,)

    for cb in trainer.callbacks:
        assert not isinstance(cb, bad_types), (
            f"use_ema=False but {type(cb).__name__} was still attached"
        )
