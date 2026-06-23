"""Regression tests for SER2 (_RecurrentWrapperBase pickle) + SER3 (KerasCompatibleMLP pickle)."""

import pickle

import numpy as np
import pytest


def test_recurrent_wrapper_getstate_drops_trainer_and_cache():
    """SER2: __getstate__ must null the live trainer + _prediction_cache and move the model to
    CPU so the generic dill/pickle path can round-trip the wrapper without a live Trainer."""
    pytest.importorskip("lightning")
    torch = pytest.importorskip("torch")
    from mlframe.training.neural.recurrent_dataset_helpers import _RecurrentWrapperBase

    w = _RecurrentWrapperBase()
    # Simulate a fitted wrapper: live (unpicklable-via-safe-loader) trainer + a populated cache + a torch model.
    import lightning as L

    w.trainer = L.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)
    w._prediction_cache = {b"key": np.zeros(3)}
    w.model = torch.nn.Linear(2, 1)

    restored = pickle.loads(pickle.dumps(w))

    assert restored.trainer is None
    assert restored._prediction_cache == {}
    assert restored.model is not None
    assert next(restored.model.parameters()).device.type == "cpu"


def test_keras_compatible_mlp_roundtrip_preserves_predictions():
    """SER3: a fitted KerasCompatibleMLP must pickle via get_config()+get_weights() and predict
    identically after a round-trip."""
    pytest.importorskip("tensorflow")
    from mlframe.training.neural.keras_compat import KerasCompatibleMLP

    rng = np.random.default_rng(0)
    X = rng.random((40, 4)).astype(np.float32)
    y = (X.sum(axis=1) + rng.normal(0, 0.01, 40)).astype(np.float32)

    est = KerasCompatibleMLP(num_layers=1, num_neurons=8, epochs=2, batch_size=16, validation_split=0.0)
    est.fit(X, y)
    before = est.predict(X)

    restored = pickle.loads(pickle.dumps(est))
    after = restored.predict(X)

    assert restored.model_ is not None
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
