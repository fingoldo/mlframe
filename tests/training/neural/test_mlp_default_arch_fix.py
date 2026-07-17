"""Locks the 2026-05-26 MLP default-arch fix.

Observed failure: on a 4.1M-row, 206-feature TVT regression
(MTTR/MTTS ~= 11497), Ridge produced RMSE=11.18 / R^2=1.00 while the
default PyTorchLightningRegressor MLP collapsed to a bimodal prediction
pattern at the train-range rails (RMSE=3641, R^2=-30.84). Root cause:

* ``use_batchnorm=False`` + ``use_layernorm=False`` (norm=none) +
  LeakyReLU + Adam + kaiming_normal init -> inner pre-activations
  saturated.
* ``output_activation='tanh_train_range'`` was only enabled via the
  extreme-AR gate, hard-capping saturated outputs to +-1 in
  standardised space and producing bimodal destandardised predictions.

The fix flips two defaults in ``trainer._configure_mlp_params``:

1. ``use_batchnorm=True`` (BN normalises per-feature ACROSS the batch
   -- exactly the axis that LN destroys; safe for tabular regression).
2. ``output_activation='tanh_train_range'`` (zero-config hard cap;
   scale + center auto-derived from y_train in
   ``neural.base._fit_inner_network``).

These tests monkey-patch the lightning-estimator class in the trainer
module with a recorder, drive ``_configure_mlp_params`` with a minimal
stub configs object, and assert the ``network_params`` kwarg the
configurator passes to the estimator carries the corrected defaults.
This sidesteps the heavyweight ``MLP_GENERAL_PARAMS`` plumbing while
still locking behaviour (not source-string regex).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


pytest.importorskip("pytorch_lightning")


class _RecorderEstimator:
    """Stand-in for ``PytorchLightningRegressor`` /
    ``PytorchLightningClassifier``. Captures the ``network_params``
    kwarg it was constructed with so the test can introspect."""

    last_network_params: dict | None = None

    def __init__(self, network_params=None, **kwargs):
        type(self).last_network_params = dict(network_params or {})
        self.network_params = dict(network_params or {})

    # Stubs to satisfy any downstream call by the configurator / TTR
    # wrapper. The configurator only constructs + wraps with
    # ``_TTRWithEvalSetScaling`` for regression -- TTR's __init__ is
    # tolerant of any "regressor" object.
    def fit(self, X, y=None, **kwargs):
        """Fit."""
        return self

    def predict(self, X):
        """Predict."""
        import numpy as np

        return np.zeros(len(X))


def _drive_configurator(use_regression: bool, monkeypatch) -> dict:
    """Patch the neural-estimator classes in the trainer module to the
    recorder, drive ``_configure_mlp_params`` with a minimal stub
    configs object, and return the captured ``network_params`` dict."""
    from mlframe.training import trainer

    # Reset recorder between calls.
    _RecorderEstimator.last_network_params = None

    # The configurator pulls the estimator classes via
    # ``_get_neural_components()``; patch that to return the recorder
    # for both regression + classification heads. Arch enum stays real
    # (Declining is referenced by name in the dict literal).
    from mlframe.training.neural.flat import MLPNeuronsByLayerArchitecture

    monkeypatch.setattr(
        trainer,
        "_get_neural_components",
        lambda: (MLPNeuronsByLayerArchitecture, _RecorderEstimator, _RecorderEstimator),
    )

    configs = SimpleNamespace(MLP_GENERAL_PARAMS={})
    metamodel_func = lambda m: m

    trainer._configure_mlp_params(
        configs=configs,
        config_params={},
        use_regression=use_regression,
        metamodel_func=metamodel_func,
        target_type=None,
        n_train=200_000,  # > _SMALL_DATA_NLAYERS_AUTO_TUNE_THRESHOLD
    )
    captured = _RecorderEstimator.last_network_params
    assert captured is not None, "configurator did not instantiate the estimator"
    return captured


class TestMlpDefaultArchFix:
    """Groups tests covering mlp default arch fix."""
    def test_use_batchnorm_default_is_true(self, monkeypatch) -> None:
        """Use batchnorm default is true."""
        params = _drive_configurator(use_regression=True, monkeypatch=monkeypatch)
        assert params.get("use_batchnorm") is True, "use_batchnorm must default to True after the 2026-05-26 fix"

    def test_output_activation_default_is_tanh_train_range(self, monkeypatch) -> None:
        """Output activation default is tanh train range."""
        params = _drive_configurator(use_regression=True, monkeypatch=monkeypatch)
        assert params.get("output_activation") == "tanh_train_range", (
            "output_activation must default to 'tanh_train_range' so the regression head is hard-capped without the extreme-AR gate"
        )

    def test_use_layernorm_stays_false(self, monkeypatch) -> None:
        """Wave 2026-05-21 rationale is unchanged: LN_in destroys
        inter-row absolute-scale signal on tabular regression."""
        params = _drive_configurator(use_regression=True, monkeypatch=monkeypatch)
        assert params.get("use_layernorm") is False

    def test_user_override_wins_for_batchnorm(self, monkeypatch) -> None:
        """Caller-supplied ``mlp_kwargs['network_params']`` must take
        precedence -- back-compat for users who already opted out."""
        from mlframe.training import trainer

        _RecorderEstimator.last_network_params = None
        from mlframe.training.neural.flat import MLPNeuronsByLayerArchitecture

        monkeypatch.setattr(
            trainer,
            "_get_neural_components",
            lambda: (MLPNeuronsByLayerArchitecture, _RecorderEstimator, _RecorderEstimator),
        )
        configs = SimpleNamespace(MLP_GENERAL_PARAMS={})
        trainer._configure_mlp_params(
            configs=configs,
            config_params={"mlp_kwargs": {"network_params": {"use_batchnorm": False}}},
            use_regression=True,
            metamodel_func=lambda m: m,
            target_type=None,
            n_train=200_000,
        )
        assert _RecorderEstimator.last_network_params.get("use_batchnorm") is False
