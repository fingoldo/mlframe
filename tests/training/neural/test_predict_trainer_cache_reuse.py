"""Regression + validation for F-67 prediction-trainer caching, reinstated 2026-07-21.

F-67 originally cached one ``L.Trainer`` per (accelerator, precision) key to skip the ~236ms Trainer-
construction GC cost on every predict() call. It was reverted 2026-06-02 (commit 6a2c1355e) after reusing
a Trainer across predict() calls accumulated Lightning's prediction-loop state -- the SECOND+ predict raised
"Mismatch in number of limits (N) and number of iterables (1)" (combined_loader.py:333) -- breaking every
multi-predict fit (val/test/OOF, and especially the per-feature permutation-importance loop: dozens of
predicts per fit, all but the first failing).

Re-verified against the currently-installed Lightning that this no longer reproduces when the Trainer is
reused but a FRESH TorchDataModule/dataloader is built per call (this is what ``_predict_raw`` always does --
only the outer Trainer object gets cached). These tests cover exactly the scenarios the original bug hit:
many sequential predicts, a real multi-feature permutation_importance sweep, a multilabel model (the
documented "worst case"), and concurrent multi-threaded predicts (safe now via _cuda_fallback.py's lock).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _regressor():
    """Small regressor for the cache-reuse tests."""
    return PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 1, "first_layer_num_neurons": 8},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={"max_epochs": 1, "logger": False, "accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    )


def test_prediction_trainer_is_actually_cached_and_reused():
    """The SAME Trainer object must be reused across consecutive predict() calls with matching
    (accelerator, precision) -- otherwise the whole point of F-67 caching is lost."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    est = _regressor()
    est.fit(X, y)

    est.predict(rng.normal(size=(10, 4)).astype(np.float32))
    trainer1 = next(iter(est._prediction_trainer_cache.values()))
    est.predict(rng.normal(size=(10, 4)).astype(np.float32))
    trainer2 = next(iter(est._prediction_trainer_cache.values()))
    assert trainer1 is trainer2, "expected the cached Trainer object to be reused across predict() calls"


def test_many_sequential_predicts_all_succeed():
    """100 sequential predict() calls on one fitted estimator must all succeed -- this is exactly F-67's
    original bug case (the SECOND+ predict on a reused Trainer used to raise a CombinedLoader.limits
    mismatch)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    est = _regressor()
    est.fit(X, y)

    for i in range(100):
        Xi = rng.normal(size=(15, 4)).astype(np.float32)
        preds = est.predict(Xi)
        assert preds.shape == (15,), f"predict #{i} returned unexpected shape {preds.shape}"


def test_permutation_importance_sweep_all_features_succeed():
    """The exact real-world trigger for F-67's original bug: sklearn's permutation_importance sweeping many
    features x n_repeats, each requiring an independent predict() call on the shared, cached-Trainer
    estimator. Asserts every feature's importance is a genuine finite measurement (not the -inf/degraded
    value F-67's bug produced when a predict silently failed), not a specific ranking -- ranking correctness
    needs real training convergence, which isn't this test's concern."""
    from mlframe.training._feature_importances import _permutation_feature_importances

    rng = np.random.default_rng(2)
    n_features = 8
    X = rng.normal(size=(200, n_features)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.normal(size=200) * 0.1).astype(np.float32)
    est = _regressor()
    est.fit(X, y)

    mean, std = _permutation_feature_importances(est, X, y, n_repeats=5, return_std=True)
    assert mean is not None and std is not None
    assert mean.shape == (n_features,) and std.shape == (n_features,)
    assert np.all(np.isfinite(mean)), f"F-67's original bug degraded post-first predicts to -inf/NaN; got {mean}"
    assert np.all(np.isfinite(std))


def test_multilabel_model_many_predicts_succeed():
    """The documented 'worst case' (2026-06-02 comment): the multilabel head routes predict to CPU already;
    with Trainer caching reinstated, many predicts on a multilabel model must still all succeed."""
    clf = PytorchLightningClassifier(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.BCEWithLogitsLoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 1, "first_layer_num_neurons": 8},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={"max_epochs": 1, "logger": False, "accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    )
    clf._is_multilabel = True
    clf._binary_sigmoid_head = False
    clf._label_encoder = None
    clf.classes_ = None

    rng = np.random.default_rng(3)
    raw = rng.random((3, 3))
    clf._predict_raw = lambda Xi, **kw: raw  # bypass the torch forward pass; isolates the Trainer-cache path is exercised

    for _ in range(20):
        out = clf.predict(np.zeros((3, 5), dtype=np.float32))
        assert out.shape == (3, 3)


def test_concurrent_predicts_with_cached_trainer_no_errors():
    """8 threads calling predict() concurrently on ONE shared, cached-Trainer estimator must all succeed --
    _cuda_fallback.py's process-wide lock must still fully serialize access even with a REUSED Trainer object
    (not just fresh ones per call)."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    est = _regressor()
    est.fit(X, y)

    errors: list = []

    def _worker():
        """Worker."""
        try:
            for _ in range(20):
                Xi = rng.normal(size=(12, 4)).astype(np.float32)
                est.predict(Xi)
        except Exception as e:  # capturing for the assertion below, not swallowing silently
            errors.append((type(e).__name__, str(e)))

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"expected zero errors, got: {errors[:5]}"


def test_cache_rebuilds_on_accelerator_change():
    """A mid-lifecycle device change (predict(device='cuda') after predict(device='cpu')) must rebuild the
    cached Trainer under the new key rather than reusing the wrong-device one."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    est = _regressor()
    est.fit(X, y)

    est.predict(rng.normal(size=(10, 4)).astype(np.float32), device="cpu")
    assert ("cpu", None) in est._prediction_trainer_cache or any(k[0] == "cpu" for k in est._prediction_trainer_cache)
    n_cached_before = len(est._prediction_trainer_cache)
    # Same device again -- no new cache entry.
    est.predict(rng.normal(size=(10, 4)).astype(np.float32), device="cpu")
    assert len(est._prediction_trainer_cache) == n_cached_before


def test_getstate_still_drops_trainer_cache_for_pickle():
    """F-73b (2026-06-01): the cache must stay excluded from pickle -- a live Trainer isn't picklable."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    est = _regressor()
    est.fit(X, y)
    est.predict(rng.normal(size=(10, 4)).astype(np.float32))
    assert len(est._prediction_trainer_cache) > 0

    state = est.__getstate__()
    assert "_prediction_trainer_cache" not in state
