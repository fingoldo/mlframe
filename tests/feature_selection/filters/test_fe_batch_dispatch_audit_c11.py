"""Gate fallbacks of the FE batch dispatcher and the auto-escalation are logged at WARNING; a mismatched target length is refused."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_batch_dispatch as D
from mlframe.feature_selection.filters import _fe_auto_escalation as E
from mlframe.feature_selection.filters import _fe_auto_escalation_proposers as P


def test_ktc_choice_failure_is_logged_at_warning(monkeypatch, caplog):
    """A failing kernel-tuning lookup falls back to None AND says so at WARNING."""
    import mlframe.feature_selection.filters._fe_batch_dispatch_ktc as ktc

    def boom(*a, **k):
        """Raise like a broken tuning cache."""
        raise RuntimeError("ktc broke")

    monkeypatch.setattr(ktc, "fe_batch_use_device", boom)
    with caplog.at_level(logging.WARNING, logger=D.logger.name):
        assert D._ktc_backend_choice(100, 10) is None
    assert any("ktc broke" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_gpu_choice_without_cuda_is_logged_at_warning(monkeypatch, caplog):
    """A forced GPU backend on a host without CUDA downgrades to CPU with a WARNING."""
    monkeypatch.setenv("MLFRAME_FE_VRAM_BACKEND", "gpu")
    monkeypatch.setattr(D, "_cuda_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=D.logger.name):
        assert D.choose_fe_batch_backend(1000, 10) == "cpu"
    assert any(r.levelno == logging.WARNING and "no usable CUDA" in r.getMessage() for r in caplog.records)


def test_fe_batch_mi_rejects_target_of_other_length():
    """A y shorter than the candidate matrix would be read out of bounds by the kernels: refuse it."""
    X = np.zeros((100, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="y_codes has 90 rows"):
        D.fe_batch_mi(X, np.zeros(90, dtype=np.int64), backend="cpu")


def test_underdelivery_pair_failure_is_logged_at_warning(caplog):
    """A pair that makes the trigger raise is skipped with a WARNING instead of a debug line."""

    class _Self:
        """Minimal estimator stand-in."""

        quantization_nbins = 8
        feature_names_in_ = np.array(["a", "b"])

    class _Frame(dict):
        """Dict frame with a length."""

        def __len__(self):
            """Row count."""
            return 50

    X = _Frame({"a": np.arange(50.0), "b": np.arange(50.0)})
    with caplog.at_level(logging.WARNING, logger=E.logger.name):
        out = E.find_underdelivering_pairs(
            _Self(), prospective_pairs=[((0, 1), "not-a-float")], prospective_additions={}, X=X, cols=["a", "b"],
            classes_y=np.arange(50) % 2, done=set(),
        )
    assert out == []
    assert any(r.levelno == logging.WARNING and "find_underdelivering_pairs" in r.getMessage() for r in caplog.records)


def test_pair_basis_failure_is_logged_at_warning(monkeypatch, caplog):
    """A basis whose ALS solve raises is skipped with a WARNING."""
    import mlframe.feature_selection.filters.hermite_fe as H

    def boom(*a, **k):
        """Raise like a failing ALS solve."""
        raise RuntimeError("als broke")

    monkeypatch.setattr(H, "warm_start_als_seed", boom)
    blk = (np.ones((4, 2)), np.ones((4, 2)), np.ones(4))
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        basis, _ = P._best_pair_basis({"chebyshev": blk}, {"chebyshev": blk}, np.ones(4), np.ones(4), np.ones(4), np.ones(4))
    assert basis is None
    assert any(r.levelno == logging.WARNING and "als broke" in r.getMessage() for r in caplog.records)
