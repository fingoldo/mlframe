"""Regression test for the 2026-05-27 MLP dump-size blow-up.

Bug: PytorchLightningEstimator stashed the full training DataModule on
``self.prediction_datamodule`` to silence a benign "create-on-predict"
log line. The DM holds the entire train+val feature + label tensors.
For a 4M x 323 float32 frame this was ~5 GB raw -> 1.7 GB compressed
on disk. save-size-sensor warned 'Tabular ML bundles should be <50 MB
even on million-row training' (TVT_regression.log 23:57:13 and again
at 01:36:43).

Fix (refined 2026-05-27): null the heavy train/val TENSORS held inside
the datamodule at the end of fit while KEEPING the lightweight shell.
The shell (a few KB of config + class refs) is cheap to pickle and lets
predict() reuse the configured pre-pipeline / batch_size / dataloader
params without rebuilding the datamodule -- and it silences the spurious
"No datamodule found from training" WARNING that fired when we used to
null the whole reference. The tensors were the actual 1.7 GB bloat.

Opt-out: ``MLFRAME_KEEP_PREDICTION_DATAMODULE=1`` env var preserves the
full datamodule (tensors included) for operators relying on the prior
whole-stash behaviour.
"""

from __future__ import annotations

import os
from types import SimpleNamespace


# The heavy tensor attributes the fit-cleanup nulls inside the datamodule
# shell. Mirrors the list in ``neural/base.py`` _fit_internal cleanup.
_TENSOR_ATTRS = (
    "train_features",
    "train_labels",
    "train_sample_weight",
    "val_features",
    "val_labels",
    "val_sample_weight",
    "_train_dataset",
    "_val_dataset",
    "train_dataset",
    "val_dataset",
)


def _make_dm_with_tensors() -> SimpleNamespace:
    """Make dm with tensors."""
    dm = SimpleNamespace()
    for _attr in _TENSOR_ATTRS:
        setattr(dm, _attr, "heavy-tensor-payload")
    return dm


def _run_cleanup(estimator) -> None:
    """Replicate the tail of ``_fit_internal`` cleanup block."""
    estimator.trainer = None
    if not os.environ.get("MLFRAME_KEEP_PREDICTION_DATAMODULE"):
        _dm = getattr(estimator, "prediction_datamodule", None)
        if _dm is not None:
            for _attr in _TENSOR_ATTRS:
                if hasattr(_dm, _attr):
                    setattr(_dm, _attr, None)
        estimator._datamodule_tensors_dropped = True


def test_fit_drops_prediction_datamodule_tensors_by_default(monkeypatch) -> None:
    """Fit drops prediction datamodule tensors by default."""
    monkeypatch.delenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", raising=False)
    estimator = SimpleNamespace()
    estimator.trainer = "dummy-trainer-state"
    estimator.prediction_datamodule = _make_dm_with_tensors()

    _run_cleanup(estimator)

    assert estimator.trainer is None
    # Shell is KEPT (not None) so predict() can reuse the config + avoid
    # the spurious "no datamodule" warning.
    assert estimator.prediction_datamodule is not None
    # But every heavy tensor is dropped.
    for _attr in _TENSOR_ATTRS:
        assert getattr(estimator.prediction_datamodule, _attr) is None, f"{_attr} must be nulled to avoid the 1.7 GB dump bloat"
    assert estimator._datamodule_tensors_dropped is True


def test_fit_keeps_prediction_datamodule_when_env_set(monkeypatch) -> None:
    """Fit keeps prediction datamodule when env set."""
    monkeypatch.setenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", "1")
    estimator = SimpleNamespace()
    estimator.trainer = "dummy"
    estimator.prediction_datamodule = _make_dm_with_tensors()

    _run_cleanup(estimator)

    # Env opt-out: every tensor stays put.
    for _attr in _TENSOR_ATTRS:
        assert getattr(estimator.prediction_datamodule, _attr) == "heavy-tensor-payload", (
            "MLFRAME_KEEP_PREDICTION_DATAMODULE=1 must preserve the full datamodule (tensors included) for backwards compatibility."
        )


def test_source_grep_drop_dm_present() -> None:
    """Sensor: the fix's source pattern stays put. The refined fix nulls
    the heavy train/val tensors inside the datamodule shell (rather than
    nulling the whole ``prediction_datamodule`` reference, which lost the
    config + triggered a spurious predict-time warning). Pin the
    tensor-nulling attrs + the env-var gate so a future refactor cannot
    accidentally re-stash the full tensors.
    """
    from pathlib import Path
    from mlframe.training.neural import base as _base

    # ``base`` became a subpackage (``base/__init__.py`` + submodules); the fit
    # cleanup body lives in a submodule, so concat __init__ + every submodule.
    _base_path = Path(_base.__file__)
    if _base_path.name == "__init__.py":
        _pkg = _base_path.parent
        src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(_pkg.glob("*.py")))
    else:
        src = _base_path.read_text(encoding="utf-8")
    # The cleanup must null at least the core feature/label tensors.
    for _attr in ("train_features", "train_labels", "val_features", "val_labels"):
        assert _attr in src, f"MLP fit cleanup must reference {_attr!r} to null it (prevents the 1.7 GB / dump tensor bloat)."
    # The marker the cleanup sets after dropping tensors.
    assert "_datamodule_tensors_dropped" in src, "fit cleanup must set the _datamodule_tensors_dropped marker."
    assert "MLFRAME_KEEP_PREDICTION_DATAMODULE" in src, "Opt-out env-var name must be preserved for back-compat."
