"""TRAINING_LOOSE_B-4 regression: a BROKEN neural install must not retry the import on every call.

Pre-fix, ``_get_neural_components`` only cached the success path (``MLPNeuronsByLayerArchitecture`` etc.
staying non-None). A broken (not merely absent) ``mlframe.training.neural`` install raises ``ImportError``
every call, so the module-level globals stay ``None`` forever and every subsequent call retries the same
documented 30-180s cold import chain.
"""

from __future__ import annotations

import builtins

import mlframe.training._model_factories as _mf


def test_broken_neural_import_is_cached_after_first_failure(monkeypatch):
    """A first ``ImportError`` from ``mlframe.training.neural`` must not trigger a second import attempt."""
    monkeypatch.setattr(_mf, "MLPNeuronsByLayerArchitecture", None)
    monkeypatch.setattr(_mf, "PytorchLightningRegressor", None)
    monkeypatch.setattr(_mf, "PytorchLightningClassifier", None)
    monkeypatch.setattr(_mf, "_NEURAL_IMPORT_FAILED", False)

    real_import = builtins.__import__
    call_count = 0

    def _spying_import(name, *args, **kwargs):
        """Counts attempts to import mlframe.training.neural and raises ImportError for them."""
        nonlocal call_count
        if name == "mlframe.training.neural" or name.startswith("mlframe.training.neural"):
            call_count += 1
            raise ImportError("simulated broken neural install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _spying_import)

    first = _mf._get_neural_components()
    second = _mf._get_neural_components()

    assert first == (None, None, None)
    assert second == (None, None, None)
    assert call_count == 1, f"expected exactly one import attempt after caching the failure, saw {call_count}"
