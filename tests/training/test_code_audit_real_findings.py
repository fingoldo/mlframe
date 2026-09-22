"""Behaviour pinned for the code-audit findings that were real defects rather than false positives.

Each case is a place where a failure used to disappear: a swallowed predict error that silently changed every prediction,
a broad ``except`` that turned any bug into "not recurrent", an ``or`` that re-ran a measurement already taken.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.composite.estimator._smearing import N_SMEAR_QUANTILES, residual_quantiles


class _RaisingModel:
    """An estimator whose predict fails, standing in for a model that cannot score the residual sample."""

    def predict(self, X):
        """Always fails."""
        raise ValueError("feature names mismatch")


class TestSmearingFailureIsVisible:
    """Smearing turning itself off changes every prediction of the model, so it must be said."""

    def test_predict_failure_warns_and_returns_none(self, caplog):
        """The defect: ``except Exception: return None`` dropped smearing with no signal at all."""
        from mlframe.utils.log_throttle import reset_throttle_counts

        reset_throttle_counts("smearing_residual_predict_failed")
        X = np.zeros((N_SMEAR_QUANTILES * 8, 2))
        t = np.ones(N_SMEAR_QUANTILES * 8)
        with caplog.at_level(logging.WARNING):
            assert residual_quantiles(_RaisingModel(), X, t) is None
        assert any("smearing disabled" in r.getMessage() for r in caplog.records)


class TestUnknownTransformIsTheOnlyExcusedFailure:
    """An unregistered transform name is a legitimate "no"; any other exception is a bug and must propagate."""

    def test_unknown_transform_is_not_recurrent(self):
        """The narrowed handler still answers False for a name nothing registered."""
        from mlframe.training.composite.ensemble import _spec_is_recurrent

        assert _spec_is_recurrent({"transform_name": "no_such_transform_xyz"}) is False

    def test_other_errors_propagate_from_recurrence_probe(self, monkeypatch):
        """The defect: ``except Exception`` turned a registry bug into a silent "not recurrent"."""
        import mlframe.training.composite.ensemble as ens

        def _boom(name):
            raise RuntimeError("registry corrupted")

        monkeypatch.setattr(ens, "get_transform", _boom)
        with pytest.raises(RuntimeError):
            ens._spec_is_recurrent({"transform_name": "log_y"})

    def test_other_errors_propagate_from_waic_probe(self, monkeypatch):
        """Same narrowing for the WAIC additivity probe."""
        from types import SimpleNamespace

        import mlframe.training.composite.discovery._tiny_rerank_waic as waic

        assert waic._additive_in_t(SimpleNamespace(transform_name="no_such_transform_xyz")) is False

        def _boom(name):
            raise RuntimeError("registry corrupted")

        monkeypatch.setattr(waic, "get_transform", _boom)
        with pytest.raises(RuntimeError):
            waic._additive_in_t(SimpleNamespace(transform_name="log_y"))


class TestFittedPipelineProbe:
    """Unfitted means NotFittedError; an object sklearn cannot inspect at all is reported, not silently routed past."""

    def test_unfitted_sklearn_transformer_is_unfitted(self):
        """The ordinary placeholder case stays a quiet False."""
        from sklearn.preprocessing import StandardScaler

        from mlframe.training.core._predict_composite_routing import _is_fitted_pipeline

        assert _is_fitted_pipeline(StandardScaler()) is False

    def test_fitted_transformer_is_fitted(self):
        """A fitted transformer reads as fitted."""
        from sklearn.preprocessing import StandardScaler

        from mlframe.training.core._predict_composite_routing import _is_fitted_pipeline

        assert _is_fitted_pipeline(StandardScaler().fit(np.zeros((3, 1)))) is True

    def test_non_estimator_warns(self, caplog):
        """The defect: a non-estimator was treated as an unfitted placeholder with only a debug line."""
        from mlframe.training.core._predict_composite_routing import _is_fitted_pipeline

        with caplog.at_level(logging.WARNING):
            assert _is_fitted_pipeline(object()) is False
        assert any("not an sklearn estimator" in r.getMessage() for r in caplog.records)


class TestHonestOofPrepassContract:
    """``None`` means "not measured"; ``{}`` means "measured, nothing scorable" and must not trigger a second measurement."""

    def test_skipped_prepass_returns_none(self):
        """When the sweep must run anyway, the prepass reports that it did not measure."""
        from mlframe.training.composite.discovery._tiny_rerank_honest import _honest_oof_prepass

        assert _honest_oof_prepass(None, None, "y", [], [], None, None, will_run=False, per_bin_enabled=False, use_wilcoxon=False) is None
        assert _honest_oof_prepass(None, None, "y", [], [], None, None, will_run=True, per_bin_enabled=True, use_wilcoxon=False) is None


def test_post_processed_model_probe_is_the_composite_wrapper_probe():
    """The two identical predicates are now one, so they cannot drift apart."""
    from mlframe.training.core._phase_persist_post import _is_post_processed_model
    from mlframe.training.core._predict_composite_routing import is_composite_wrapper

    assert _is_post_processed_model is is_composite_wrapper
