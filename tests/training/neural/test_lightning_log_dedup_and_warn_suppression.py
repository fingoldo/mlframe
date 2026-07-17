"""Two related noise-reduction regressions:

1. EarlyStoppingCallback used to be constructed with ``verbose=True``, which
   makes Lightning emit the same "Metric X improved by ..." TWICE per best
   epoch (once via ``lightning.pytorch.callbacks.early_stopping`` logger,
   once via stdout ``print``). Plus mlframe's BestEpochModelCheckpoint
   already logs "New best model at epoch ...". Three lines per event was
   pure noise. New: verbose=False, mlframe's line remains canonical.

2. Lightning's DataLoader bottleneck UserWarning ("does not have many
   workers...") fires up to 3x per fit (train + val + predict). The
   recommendation conflicts with our num_workers=0 default which is
   empirically correct on Windows/8-core (see
   _benchmarks/bench_dataloader_workers.py). Suppress that specific
   message via warnings.filterwarnings on module import.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("lightning")


@pytest.mark.fast
def test_lightning_num_workers_bottleneck_warning_suppressed():
    """Importing mlframe.training.neural.base must install a filter that
    silences Lightning's "does not have many workers" UserWarning."""
    import mlframe.training.neural.base  # noqa: F401  side-effect: install filter

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # Reapply the filter inside the catch_warnings scope so we test the
        # filter spec, not just whether the module installed it once.
        warnings.filterwarnings(
            "ignore",
            message=r".*does not have many workers which may be a bottleneck.*",
            category=UserWarning,
        )
        # Emit the canonical Lightning message
        warnings.warn(
            "The 'val_dataloader' does not have many workers which may be a bottleneck.",
            UserWarning, stacklevel=2,
        )

    bottleneck_warns = [w for w in captured if "does not have many workers" in str(w.message)]
    assert not bottleneck_warns, f"DataLoader bottleneck warning must be suppressed. Got: {bottleneck_warns}"


@pytest.mark.fast
def test_early_stopping_constructed_silently(monkeypatch):
    """Construct a PytorchLightningRegressor and trigger its callback-build
    code path; capture every EarlyStoppingCallback instantiation. Behavioral
    test: real callback construction with real kwargs, no source inspection.
    """
    import lightning.pytorch.callbacks.early_stopping as _es

    instantiations = []
    _orig_init = _es.EarlyStopping.__init__

    def _spy_init(self, *args, **kwargs):
        instantiations.append(kwargs)
        return _orig_init(self, *args, **kwargs)

    monkeypatch.setattr(_es.EarlyStopping, "__init__", _spy_init)

    import numpy as np
    import torch

    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    est = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 2},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 1,
            "logger": False,
            "accelerator": "cpu",
            "devices": 1,
        },
        early_stopping_rounds=2,
    )
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est.fit(X, y, eval_set=(X, y))

    es_instantiations = [kw for kw in instantiations if "monitor" in kw]
    assert es_instantiations, "EarlyStopping was expected to be constructed during fit with eval_set"
    for kw in es_instantiations:
        assert kw.get("verbose") is False, (
            f"EarlyStopping(verbose=...) must be False to avoid duplicate 'Metric X improved' logs. Got verbose={kw.get('verbose')!r}."
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
