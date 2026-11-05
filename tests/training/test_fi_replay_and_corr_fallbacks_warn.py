"""Fallbacks that change the answer are logged at WARNING, not debug.

A failed replay of the pre-pipeline gives importances on unscaled input, and a failed GPU correlation sends the stacking down the numpy path.
Both used to log at debug only.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

from mlframe.models.ensembling import _correlation
from mlframe.training import _feature_importances as fi


class _BrokenPipeline:
    """A stamped pre-pipeline whose transform fails on the frame it is handed."""

    def transform(self, X):
        """Always fail, as a pipeline fitted on other columns would."""
        raise RuntimeError("columns differ")


class _Model:
    """A model that carries the stamped pre-pipeline the way the fit path leaves it."""

    _mlframe_pre_pipeline = _BrokenPipeline()


def test_failed_pre_pipeline_replay_warns_that_importances_use_unscaled_input(caplog):
    """The replay failure is a WARNING that names the consequence for the importances."""
    with caplog.at_level(logging.WARNING, logger=fi.logger.name):
        out = fi._net_input_frame(_Model(), np.zeros((3, 2)))
    assert out.shape == (3, 2)
    assert any(r.levelno == logging.WARNING and "unscaled" in r.getMessage() for r in caplog.records)


def test_failed_gpu_correlation_warns_and_still_returns_the_numpy_result(monkeypatch, caplog):
    """A raising cupy path logs a WARNING and the numpy correlation comes back."""
    broken = types.SimpleNamespace(asarray=lambda M: (_ for _ in ()).throw(RuntimeError("device lost")))
    monkeypatch.setitem(sys.modules, "cupy", broken)
    monkeypatch.setattr(_correlation, "_has_cupy", lambda: True)
    M = np.random.default_rng(0).normal(size=(60, 40))
    with caplog.at_level(logging.WARNING, logger=_correlation.logger.name):
        corr = _correlation._stacked_corrcoef(M)
    np.testing.assert_allclose(corr, np.corrcoef(M))
    assert any(r.levelno == logging.WARNING and "numpy path" in r.getMessage() for r in caplog.records)
