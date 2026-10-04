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
    """Run the production fit-cleanup block that drops the datamodule tensors."""
    from mlframe.training.neural.base._base_fit_helpers import _FitCommonHelpersMixin

    _FitCommonHelpersMixin._fit_common_operators_relying_prior_whole(estimator)


def test_fit_drops_prediction_datamodule_tensors_by_default(monkeypatch) -> None:
    """Fit drops prediction datamodule tensors by default."""
    monkeypatch.delenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", raising=False)
    estimator = SimpleNamespace()
    estimator.prediction_datamodule = _make_dm_with_tensors()

    _run_cleanup(estimator)

    # Shell is KEPT (not None) so predict() can reuse the config + avoid
    # the spurious "no datamodule" warning.
    assert estimator.prediction_datamodule is not None
    # But every heavy tensor is dropped.
    assert len(_TENSOR_ATTRS) > 0
    for _attr in _TENSOR_ATTRS:
        assert getattr(estimator.prediction_datamodule, _attr) is None, f"{_attr} must be nulled to avoid the 1.7 GB dump bloat"
    assert estimator._datamodule_tensors_dropped is True


def test_fit_keeps_prediction_datamodule_when_env_set(monkeypatch) -> None:
    """Fit keeps prediction datamodule when env set."""
    monkeypatch.setenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", "1")
    estimator = SimpleNamespace()
    estimator.prediction_datamodule = _make_dm_with_tensors()

    _run_cleanup(estimator)

    # Env opt-out: every tensor stays put.
    assert len(_TENSOR_ATTRS) > 0
    for _attr in _TENSOR_ATTRS:
        assert (
            getattr(estimator.prediction_datamodule, _attr) == "heavy-tensor-payload"
        ), "MLFRAME_KEEP_PREDICTION_DATAMODULE=1 must preserve the full datamodule (tensors included) for backwards compatibility."
    assert not hasattr(estimator, "_datamodule_tensors_dropped")


def test_fit_cleanup_without_a_datamodule_only_sets_the_marker(monkeypatch) -> None:
    """An estimator that never stashed a datamodule is marked cleaned and nothing else changes."""
    monkeypatch.delenv("MLFRAME_KEEP_PREDICTION_DATAMODULE", raising=False)
    estimator = SimpleNamespace()
    _run_cleanup(estimator)
    assert estimator._datamodule_tensors_dropped is True
    assert getattr(estimator, "prediction_datamodule", None) is None
