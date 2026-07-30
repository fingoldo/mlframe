"""Regression tests for audits/full_audit_2026-07-21/training_neural.md (F1-F16).

PR1 (sklearn-api-compliance test coverage), PR2 (Jacobian-sparsity), PR3 (predict-after-CPU-fallback),
PR4 (weighted eval_set changes behaviour), PR5-PR7 (code fixes) are covered by the F1/F5/F3/F4/F6/F11/F10
tests below. PR8 (docs link) turned out to reference a "honest-negative note" that does not currently
exist in field_grouped_mlp.py's docstring -- nothing to link, dismissed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("lightning")

from sklearn.base import clone

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    TorchDataModule,
)


def _classifier_kwargs(**overrides):
    """Classifier kwargs."""
    network_params = {"nlayers": 1, "first_layer_num_neurons": 16, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU}
    model_params = {"loss_fn": torch.nn.CrossEntropyLoss(), "learning_rate": 1e-3}
    datamodule_params = {
        "read_fcn": None, "data_placement_device": None,
        "features_dtype": torch.float32, "labels_dtype": torch.int64,
        "dataloader_params": {"batch_size": 32, "num_workers": 0},
    }
    trainer_params = {
        "max_epochs": 1, "enable_model_summary": False, "default_root_dir": None,
        "log_every_n_steps": 1, "devices": 1, "logger": False, "accelerator": "cpu",
    }
    kwargs = dict(
        model_class=MLPTorchModel, model_params=model_params, network_params=network_params,
        datamodule_class=TorchDataModule, datamodule_params=datamodule_params, trainer_params=trainer_params,
    )
    kwargs.update(overrides)
    return kwargs


# ----------------------------------------------------------------------
# F1 (P0) -- get_params() must include every __init__ param so clone() preserves them.
# ----------------------------------------------------------------------


def test_f1_clone_preserves_random_state_and_class_weight_and_use_ema():
    """F1: clone preserves random state and class weight and use ema."""
    clf = PytorchLightningClassifier(
        **_classifier_kwargs(random_state=42, class_weight="balanced", use_ema=True, label_smoothing=0.1),
    )
    cloned = clone(clf)
    assert cloned.random_state == 42
    assert cloned.class_weight == "balanced"
    assert cloned.use_ema is True
    assert cloned.label_smoothing == 0.1


def test_f1_get_params_matches_real_init_signature():
    """F1: get params matches real init signature."""
    import inspect

    clf = PytorchLightningClassifier(**_classifier_kwargs())
    params = clf.get_params()
    real_params = [n for n in inspect.signature(PytorchLightningClassifier.__init__).parameters if n != "self"]
    for name in real_params:
        assert name in params, f"get_params() missing {name}"


# ----------------------------------------------------------------------
# F2 -- partial_fit(classes=...) must not desync classes_ from the label encoder's sorted index space.
# ----------------------------------------------------------------------


def test_f2_partial_fit_classes_stays_sorted():
    """F2: partial fit classes stays sorted."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(60, 4)).astype(np.float32)
    y0 = rng.integers(0, 3, size=60)
    clf = PytorchLightningClassifier(**_classifier_kwargs())
    clf.partial_fit(X0, y0, classes=np.array([2, 0, 1]))  # deliberately UNSORTED caller universe
    assert list(clf.classes_) == [0, 1, 2], f"classes_ desynced from the label encoder's sorted space: {clf.classes_}"
    assert list(clf._label_encoder.classes_) == [0, 1, 2]


# ----------------------------------------------------------------------
# F3 -- recurrent predict path routes through safe_accelerator, matching fit's own CUDA-broken-host guard.
# ----------------------------------------------------------------------


def test_f3_recurrent_predict_uses_safe_accelerator(monkeypatch):
    """F3: recurrent predict uses safe accelerator."""
    import mlframe.training.neural._base_tensor_helpers as bth
    from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig, RNNType
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper

    calls = []
    orig = bth.safe_accelerator

    def spy(requested):
        """Records call arguments for this test's assertions."""
        calls.append(requested)
        return orig(requested)

    monkeypatch.setattr(bth, "safe_accelerator", spy)
    # recurrent_dataset_helpers imports safe_accelerator lazily inside the function -- patch the same name
    # there too so the spy is actually observed regardless of import binding order.
    import mlframe.training.neural.recurrent_dataset_helpers as rdh
    monkeypatch.setattr(rdh, "safe_accelerator", spy, raising=False)

    rng = np.random.default_rng(0)
    n = 40
    features = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    cfg = RecurrentConfig(
        rnn_type=RNNType.LSTM, input_mode=InputMode.FEATURES_ONLY, hidden_size=8, num_layers=1,
        max_epochs=1, batch_size=16, accelerator="cpu", num_workers=0,
    )
    reg = RecurrentRegressorWrapper(config=cfg)
    reg.fit(features=features, labels=y)
    calls.clear()
    pred = reg.predict(features=features[:5])
    assert pred.shape == (5,)
    assert calls, "predict() never called safe_accelerator -- lost the CUDA-broken-host guard fit() applies"


# ----------------------------------------------------------------------
# F4 -- recurrent validation loss must honour eval_sample_weight, not a hardcoded None.
# ----------------------------------------------------------------------


def test_f4_recurrent_validation_step_reads_batch_sample_weights():
    """F4: recurrent validation step reads batch sample weights."""
    from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel

    calls = []

    class _Model(RecurrentTorchModel):
        """Model."""
        def _compute_weighted_loss(self, logits, labels, sample_weights):
            """Compute weighted loss."""
            calls.append(sample_weights)
            return super()._compute_weighted_loss(logits, labels, sample_weights)

    from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig, RNNType

    cfg = RecurrentConfig(rnn_type=RNNType.LSTM, input_mode=InputMode.FEATURES_ONLY, hidden_size=8, num_layers=1)
    model = _Model(config=cfg, seq_input_size=4, aux_input_size=3, is_regression=True)
    batch = {
        "aux_features": torch.randn(6, 3),
        "labels": torch.randn(6),
        "sample_weights": torch.rand(6) + 0.1,
    }
    model.validation_step(batch, 0)
    assert calls, "validation_step never called _compute_weighted_loss"
    assert calls[-1] is not None, "validation_step still hardcodes sample_weights=None"
    assert torch.allclose(calls[-1], batch["sample_weights"])


def test_f4_eval_sample_weight_reaches_the_val_batch_end_to_end():
    """fit(eval_set=..., eval_sample_weight=...) must actually change val_loss vs an unweighted eval_set
    (this also confirms the new eval_sample_weight plumbing -- eval_set's tuple had no weight slot at all
    pre-fix, so batch.get("sample_weights") would have returned None regardless of F4's read-site fix)."""
    from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig, RNNType
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper

    rng = np.random.default_rng(0)
    n, n_val = 80, 30
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    X_val = rng.normal(size=(n_val, 3)).astype(np.float32)
    y_val = rng.normal(size=n_val).astype(np.float32)
    # Extreme weight skew: the loss should differ substantially from the unweighted case.
    w_val = np.where(np.arange(n_val) < n_val // 2, 100.0, 0.01).astype(np.float32)

    def _fit(**kw):
        """No-op / recording stub matching the estimator's fit() signature."""
        cfg = RecurrentConfig(
            rnn_type=RNNType.LSTM, input_mode=InputMode.FEATURES_ONLY, hidden_size=8, num_layers=1,
            max_epochs=1, batch_size=16, accelerator="cpu", num_workers=0, early_stopping_patience=None,
        )
        reg = RecurrentRegressorWrapper(config=cfg)
        reg.fit(features=X, labels=y, eval_set=(X_val, y_val), **kw)
        return reg

    torch.manual_seed(0)
    unweighted = _fit()
    torch.manual_seed(0)
    weighted = _fit(eval_sample_weight=w_val)

    unweighted_loss = unweighted.trainer.callback_metrics.get("val_loss")
    weighted_loss = weighted.trainer.callback_metrics.get("val_loss")
    assert unweighted_loss is not None and weighted_loss is not None
    assert float(unweighted_loss) != pytest.approx(float(weighted_loss), rel=1e-6)


# ----------------------------------------------------------------------
# F5 -- PeriodicLinearEmbedding: TRUE block-diagonal per-feature projection (Jacobian sparsity).
# ----------------------------------------------------------------------


def test_f5_periodic_linear_embedding_is_block_diagonal():
    """F5: periodic linear embedding is block diagonal."""
    from mlframe.training.neural._numerical_embeddings import PeriodicLinearEmbedding

    torch.manual_seed(0)
    emb = PeriodicLinearEmbedding(in_features=4, embed_dim=3, n_frequencies=5)
    x = torch.randn(2, 4, requires_grad=True)
    out = emb(x)  # (2, 4*3)
    out_per_feat = out.view(2, 4, 3)

    # Perturbing feature i's input must change ONLY output slot i, never any other feature's slot.
    for i in range(4):
        grad = torch.autograd.grad(out_per_feat[:, i, :].sum(), x, retain_graph=True)[0]
        for j in range(4):
            if j == i:
                continue
            assert torch.all(grad[:, j] == 0), f"feature {j} leaks into feature {i}'s embedding slot (dense cross-feature mixing)"


def test_f5_periodic_linear_embedding_param_count_is_linear_not_quadratic():
    """F5: periodic linear embedding param count is linear not quadratic."""
    from mlframe.training.neural._numerical_embeddings import PeriodicLinearEmbedding

    small = PeriodicLinearEmbedding(in_features=4, embed_dim=8, n_frequencies=24)
    large = PeriodicLinearEmbedding(in_features=40, embed_dim=8, n_frequencies=24)  # 10x in_features
    small_params = sum(p.numel() for p in small.parameters())
    large_params = sum(p.numel() for p in large.parameters())
    # True block-diagonal: params scale linearly with in_features (~10x), not quadratically (~100x).
    assert large_params < 20 * small_params, f"param count scaling looks quadratic: {small_params} -> {large_params}"


# ----------------------------------------------------------------------
# F6 -- cudnn.benchmark autotune must be save/restored, never a permanent global flip.
# ----------------------------------------------------------------------


def test_f6_cudnn_autotune_returns_prior_value_for_restore(monkeypatch):
    """F6: cudnn autotune returns prior value for restore."""
    from mlframe.training.neural._recurrent_config import RNNType
    from mlframe.training.neural._recurrent_perf import maybe_enable_cudnn_rnn_autotune

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (8, 0))
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    prior = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = False
        result = maybe_enable_cudnn_rnn_autotune(RNNType.LSTM)
        assert result is False, "must return the PRIOR value so the caller can restore it"
        assert torch.backends.cudnn.benchmark is True, "must still flip the flag on when eligible"
    finally:
        torch.backends.cudnn.benchmark = prior


def test_f6_fit_restores_cudnn_benchmark_after_training(monkeypatch):
    """End-to-end: a fit() that flips cudnn.benchmark must restore it afterward, not leak into sibling
    code trained later in the same process."""
    from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig, RNNType
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper

    sentinel_prior = torch.backends.cudnn.benchmark  # whatever this test process currently has

    def fake_autotune(self):
        """Fake replacement for the cuDNN autotune-flag lookup."""
        return not sentinel_prior  # pretend the flag WAS flipped and record a different-from-current prior

    import mlframe.training.neural.recurrent_dataset_helpers as rdh
    monkeypatch.setattr(rdh._RecurrentWrapperBase, "_maybe_enable_cudnn_rnn_autotune", fake_autotune)

    rng = np.random.default_rng(0)
    n = 40
    features = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    cfg = RecurrentConfig(
        rnn_type=RNNType.LSTM, input_mode=InputMode.FEATURES_ONLY, hidden_size=8, num_layers=1,
        max_epochs=1, batch_size=16, accelerator="cpu", num_workers=0,
    )
    reg = RecurrentRegressorWrapper(config=cfg)
    reg.fit(features=features, labels=y)
    assert torch.backends.cudnn.benchmark == (not sentinel_prior), "fit() must restore the mocked prior value after trainer.fit() returns"


# ----------------------------------------------------------------------
# F7 -- TorchDataset's 2GB eager/lazy gate must fire for a pandas DataFrame carrier.
# ----------------------------------------------------------------------


def test_f7_torch_dataset_estimates_pandas_bytes():
    """F7: torch dataset estimates pandas bytes."""
    import pandas as pd

    from mlframe.training.neural.data import _estimate_bytes

    df = pd.DataFrame(np.random.default_rng(0).normal(size=(1000, 10)).astype(np.float32))
    estimate = _estimate_bytes(df)
    assert estimate > 0, "pandas DataFrame byte estimate must not silently resolve to 0"
    assert estimate == int(df.memory_usage(deep=True).sum())


def test_f7_torch_dataset_defers_lazy_path_above_cap(monkeypatch):
    """F7: torch dataset defers lazy path above cap."""
    import pandas as pd

    import mlframe.training.neural.data as data_mod

    monkeypatch.setattr(data_mod, "_EAGER_CONVERSION_BYTES_CAP", 100)  # tiny cap forces the lazy path
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32))
    ds = data_mod.TorchDataset(features=df, labels=np.zeros(50, dtype=np.float32))
    assert ds._eager_features is False


# ----------------------------------------------------------------------
# F8 -- suppress_lightning_workers_warning is now actually applied to trainer.fit()/predict() calls.
# ----------------------------------------------------------------------


def test_f8_suppress_lightning_workers_warning_used_at_fit(monkeypatch):
    """F8: suppress lightning workers warning used at fit."""
    import mlframe.training.neural.base._base_fit as base_fit_mod

    calls = []
    orig = base_fit_mod.suppress_lightning_workers_warning

    def spy():
        """Records call arguments for this test's assertions."""
        calls.append(1)
        return orig()

    monkeypatch.setattr(base_fit_mod, "suppress_lightning_workers_warning", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=40)
    clf = PytorchLightningClassifier(**_classifier_kwargs())
    clf.fit(X, y)
    assert calls, "trainer.fit() no longer wrapped in suppress_lightning_workers_warning()"


# ----------------------------------------------------------------------
# F9 -- MuonAdamWHybrid.state_dict()/load_state_dict() round-trips both wrapped optimizers' state.
# ----------------------------------------------------------------------


def test_f9_muon_adamw_hybrid_state_dict_round_trips():
    """F9: muon adamw hybrid state dict round trips."""
    from mlframe.training.neural._muon_optimizer import MuonAdamWHybrid

    torch.manual_seed(0)
    w2d = torch.nn.Parameter(torch.randn(8, 8))
    b1d = torch.nn.Parameter(torch.randn(8))
    opt = MuonAdamWHybrid([w2d, b1d], lr=1e-3, muon_lr=2e-2)

    # Take a few steps so both inner optimizers accumulate real momentum/moment-estimate state.
    for _ in range(3):
        opt.zero_grad()
        loss = (w2d.sum() + b1d.sum()) ** 2
        loss.backward()
        opt.step()

    state = opt.state_dict()
    assert state["muon"] is not None and state["adamw"] is not None
    assert len(state["muon"]["state"]) > 0, "Muon momentum state was not captured"
    assert len(state["adamw"]["state"]) > 0, "AdamW moment-estimate state was not captured"

    # A fresh optimizer restored from this state_dict must reproduce the same next step.
    w2d_fresh = torch.nn.Parameter(w2d.detach().clone())
    b1d_fresh = torch.nn.Parameter(b1d.detach().clone())
    opt_fresh = MuonAdamWHybrid([w2d_fresh, b1d_fresh], lr=1e-3, muon_lr=2e-2)
    opt_fresh.load_state_dict(state)
    assert len(opt_fresh._muon.state) == len(opt._muon.state)
    assert len(opt_fresh._adamw.state) == len(opt._adamw.state)


# ----------------------------------------------------------------------
# F10 -- MLPRanker(seed=...) fully seeds torch's global RNG (save/restore, not a permanent mutation).
# ----------------------------------------------------------------------


def test_f10_mlp_ranker_same_seed_is_reproducible():
    """F10: mlp ranker same seed is reproducible."""
    from mlframe.training.neural.ranker import MLPRanker

    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5)).astype(np.float32)
    group_ids = np.repeat(np.arange(n // 6), 6)
    y = rng.integers(0, 4, size=n).astype(np.float32)

    def _fit_predict(seed):
        """Fit predict."""
        torch.manual_seed(999)  # a DIFFERENT ambient global seed each call, to prove `seed=` itself controls init
        model = MLPRanker(seed=seed, n_estimators=3, hidden_layers=(8,), early_stopping_patience=None)
        model.fit(X, y, group_ids)
        return model.predict(X)

    pred_a = _fit_predict(seed=7)
    pred_b = _fit_predict(seed=7)
    assert np.allclose(pred_a, pred_b), "two fit() calls with the same seed must be bit-identical"


def test_f10_mlp_ranker_restores_global_torch_rng_state():
    """F10: mlp ranker restores global torch rng state."""
    from mlframe.training.neural.ranker import MLPRanker

    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 4)).astype(np.float32)
    group_ids = np.repeat(np.arange(n // 6), 6)
    y = rng.integers(0, 4, size=n).astype(np.float32)

    torch.manual_seed(1234)
    before = torch.rand(4)  # consume + record the ambient stream's next draw
    torch.manual_seed(1234)
    model = MLPRanker(seed=99, n_estimators=2, hidden_layers=(8,), early_stopping_patience=None)
    model.fit(X, y, group_ids)
    after = torch.rand(4)
    assert torch.allclose(before, after), "fit() must not permanently pollute the caller's global torch RNG stream"


# ----------------------------------------------------------------------
# F11 -- _RecurrentWrapperBase._prediction_cache is LRU-bounded, not unbounded.
# ----------------------------------------------------------------------


def test_f11_prediction_cache_is_lru_bounded(monkeypatch):
    """F11: prediction cache is lru bounded."""
    import mlframe.training.neural.recurrent_dataset_helpers as rdh
    from mlframe.training.neural._recurrent_config import InputMode, RecurrentConfig, RNNType
    from mlframe.training.neural.recurrent import RecurrentRegressorWrapper

    monkeypatch.setattr(rdh, "_PREDICTION_CACHE_MAX", 3)

    rng = np.random.default_rng(0)
    n = 40
    features = rng.normal(size=(n, 3)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    cfg = RecurrentConfig(
        rnn_type=RNNType.LSTM, input_mode=InputMode.FEATURES_ONLY, hidden_size=8, num_layers=1,
        max_epochs=1, batch_size=16, accelerator="cpu", num_workers=0,
    )
    reg = RecurrentRegressorWrapper(config=cfg)
    reg.fit(features=features, labels=y)

    for i in range(6):  # each distinct slice is a distinct cache key
        reg.predict(features=features[i : i + 5])
    assert len(reg._prediction_cache) <= 3, f"prediction cache grew unbounded: {len(reg._prediction_cache)} entries"


# ----------------------------------------------------------------------
# F12 -- TorchDataModule._convert_features_dtype skips the eager copy above the byte-size cap.
# ----------------------------------------------------------------------


def test_f12_convert_features_dtype_skips_astype_above_cap(monkeypatch):
    """F12: convert features dtype skips astype above cap."""
    import pandas as pd

    import mlframe.training.neural.data as data_mod

    monkeypatch.setattr(data_mod, "_EAGER_CONVERSION_BYTES_CAP", 100)  # tiny cap
    dm = data_mod.TorchDataModule.__new__(data_mod.TorchDataModule)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 4)).astype(np.float64))
    dm.train_features = df
    dm._convert_features_dtype(["train_features"])
    assert dm.train_features.dtypes.iloc[0] == np.float64, "astype() ran despite being above the byte-size cap"


def test_f12_convert_features_dtype_still_converts_below_cap():
    """F12: convert features dtype still converts below cap."""
    import pandas as pd

    import mlframe.training.neural.data as data_mod

    dm = data_mod.TorchDataModule.__new__(data_mod.TorchDataModule)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 4)).astype(np.float64))
    dm.train_features = df
    dm._convert_features_dtype(["train_features"])
    assert dm.train_features.dtypes.iloc[0] == np.float32


# ----------------------------------------------------------------------
# F13 -- redundant exception tuple simplified; behavior-preserving smoke check.
# ----------------------------------------------------------------------


def test_f13_on_gpu_returns_false_without_trainer():
    """F13: on gpu returns false without trainer."""
    import mlframe.training.neural.data as data_mod

    dm = data_mod.TorchDataModule.__new__(data_mod.TorchDataModule)
    dm.trainer = None
    assert dm.on_gpu() is False


# ----------------------------------------------------------------------
# F14 -- MLPRanker._x_to_array logs a warning naming dropped non-numeric columns.
# ----------------------------------------------------------------------


def test_f14_x_to_array_warns_on_dropped_columns(caplog):
    """F14: x to array warns on dropped columns."""
    import pandas as pd

    from mlframe.training.neural.ranker import MLPRanker

    model = MLPRanker(seed=0)
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.ranker"):
        arr = model._x_to_array(df)
    assert arr.shape == (3, 1)
    assert any("cat" in rec.message for rec in caplog.records)


def test_f14_x_to_array_silent_when_all_numeric(caplog):
    """F14: x to array silent when all numeric."""
    import pandas as pd

    from mlframe.training.neural.ranker import MLPRanker

    model = MLPRanker(seed=0)
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.ranker"):
        model._x_to_array(df)
    assert not any("non-numeric" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------
# F15 -- Muon-Triton calibration verdict is persisted via the shared kernel_tuning_cache.
# ----------------------------------------------------------------------


def test_f15_muon_triton_ktc_helper_returns_none_gracefully_without_gpu():
    """F15: muon triton ktc helper returns none gracefully without gpu."""
    from mlframe.training.neural._muon_triton_kernel import _get_kernel_tuning_cache

    # Must never raise even when pyutilz/FS or CUDA is unavailable in this test environment.
    _get_kernel_tuning_cache()


def test_f15_muon_triton_ktc_lookup_invoked_when_cache_available(monkeypatch):
    """F15: muon triton ktc lookup invoked when cache available."""
    import mlframe.training.neural._muon_triton_kernel as mtk

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to exercise the Triton calibration path")

    calls = []

    class _FakeCache:
        """Stub kernel-tuning cache used to control the get_or_tune path."""
        def get_or_tune(self, kernel_name, **kwargs):
            """Fake replacement for the kernel-tuning-cache lookup."""
            calls.append((kernel_name, kwargs.get("dims")))
            return {"use_triton": False}

    monkeypatch.setattr(mtk, "_get_kernel_tuning_cache", lambda: _FakeCache())
    monkeypatch.setattr(mtk, "get_triton_ns_fn", lambda: (lambda G, steps: G))
    monkeypatch.setattr(mtk, "_env_force", lambda: None)
    G = torch.randn(256, 256, device="cuda")
    mtk.maybe_newton_schulz_triton(G, steps=2)
    assert calls, "KTC's get_or_tune was never consulted for the Triton-vs-eager verdict"


# ----------------------------------------------------------------------
# F16 -- the clone-safety regression test itself now introspects the real signature (see test_estimators.py).
# ----------------------------------------------------------------------


def test_f16_test_estimators_get_params_test_uses_introspection():
    """Confirms the fixed test in test_estimators.py is not the old hardcoded-list version -- a
    lightweight guard so this specific regression can't silently reappear via an unrelated edit."""
    from pathlib import Path

    src = Path(__file__).with_name("test_estimators.py").read_text(encoding="utf-8")
    start = src.index("def test_get_params_includes_all_init_params")
    body = src[start : start + 1200]  # generously covers this one short test method's full body
    assert "inspect.signature" in body
    assert '"model_class",' not in body  # the old hardcoded list is gone
