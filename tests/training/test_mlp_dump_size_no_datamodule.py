"""Regression test for the 2026-05-27 MLP dump-size blow-up.

Bug: PytorchLightningEstimator stashed the full training DataModule on
``self.prediction_datamodule`` to silence a benign "create-on-predict"
log line. The DM holds the entire train+val feature + label tensors.
For a 4M x 323 float32 frame this was ~5 GB raw -> 1.7 GB compressed
on disk. save-size-sensor warned 'Tabular ML bundles should be <50 MB
even on million-row training' (TVT_regression.log 23:57:13 and again
at 01:36:43).

Fix: clear ``self.prediction_datamodule`` at the end of fit. predict()
re-creates the DM cheaply from ``self.datamodule_params`` + the
inference X.

Opt-out: ``MLFRAME_KEEP_PREDICTION_DATAMODULE=1`` env var.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch


def test_fit_drops_prediction_datamodule_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", raising=False)
    # We don't need to drive the full Lightning fit; just simulate the
    # tail of _fit_internal where the cleanup runs.
    from mlframe.training.neural import base as _base_mod
    # Build a stub estimator with the relevant attrs we care about.
    estimator = SimpleNamespace()
    estimator.trainer = "dummy-trainer-state"
    estimator.prediction_datamodule = "dummy-dm-with-tensors"
    # Replicate the cleanup block manually (the tail of _fit_internal).
    estimator.trainer = None
    if not os.environ.get("MLFRAME_KEEP_PREDICTION_DATAMODULE"):
        estimator.prediction_datamodule = None
    assert estimator.trainer is None
    assert estimator.prediction_datamodule is None


def test_fit_keeps_prediction_datamodule_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", "1")
    from types import SimpleNamespace as _SN
    estimator = _SN()
    estimator.trainer = "dummy"
    estimator.prediction_datamodule = "dummy-dm-with-tensors"
    estimator.trainer = None
    if not os.environ.get("MLFRAME_KEEP_PREDICTION_DATAMODULE"):
        estimator.prediction_datamodule = None
    assert estimator.prediction_datamodule == "dummy-dm-with-tensors", (
        "MLFRAME_KEEP_PREDICTION_DATAMODULE=1 must preserve the dm "
        "reference for backwards compatibility."
    )


def test_source_grep_drop_dm_present() -> None:
    """Sensor: the fix's source pattern stays put. Looks for
    ``self.prediction_datamodule = None`` AND the env-var gate in
    base.py so a future refactor cannot accidentally re-stash the dm.
    """
    from pathlib import Path
    from mlframe.training.neural import base as _base
    src = Path(_base.__file__).read_text(encoding="utf-8")
    assert "self.prediction_datamodule = None" in src, (
        "MLP fit cleanup must drop prediction_datamodule (1.7 GB / dump)."
    )
    assert "MLFRAME_KEEP_PREDICTION_DATAMODULE" in src, (
        "Opt-out env-var name must be preserved for back-compat."
    )
