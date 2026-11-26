"""Fallbacks that change a result or slow a run must say so at WARNING, not pass silently."""

import logging

import mlframe.feature_selection.filters._batch_mi_noise_gate_kernels as kernels
import mlframe.feature_selection.filters._binning_spec as binning_spec
import mlframe.feature_selection.filters._column_stats as column_stats
import mlframe.feature_selection.filters._unary_credit_logit as credit_logit


class _Estimator:
    """A bare estimator carrying the three attributes the meta fallback reads."""

    binning_scheme = "quantile"
    nbins_rule = "sturges"
    nbins_cap = 16


def test_binning_meta_falls_back_with_a_warning(monkeypatch, caplog):
    """A spec that cannot be resolved falls back to the estimator attributes and logs why."""

    def _raise(_est):
        """Stand in for a spec resolution that fails."""
        raise ValueError("no spec for this estimator")

    monkeypatch.setattr(binning_spec, "spec_from_estimator", _raise)
    with caplog.at_level(logging.WARNING, logger=column_stats.logger.name):
        meta = column_stats._binning_meta(_Estimator())
    assert meta == {"binning_scheme": "quantile", "nbins_rule": "sturges", "nbins_cap": 16}
    assert any(r.levelno == logging.WARNING and "fell back" in r.getMessage() for r in caplog.records)


def test_device_probe_failure_logs_and_runs_on_host(monkeypatch, caplog):
    """A CUDA probe that raises returns no device module and says the host null runs instead."""
    monkeypatch.delenv("MLFRAME_FE_CREDIT_GPU", raising=False)

    def _boom():
        """Stand in for a CUDA probe that fails."""
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(kernels, "cuda_available", _boom)
    with caplog.at_level(logging.WARNING):
        assert credit_logit._device_module() is None
    assert any(r.levelno == logging.WARNING and "host null runs" in r.getMessage() for r in caplog.records)
