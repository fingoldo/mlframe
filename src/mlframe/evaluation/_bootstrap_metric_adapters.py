"""Shared bootstrap-resample metric adapters wrapping ``mlframe.metrics.core``'s fast Brier/log-loss
kernels: independently duplicated between ``training.honest_diagnostics`` and its bench harness,
consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np

from mlframe.metrics.core import fast_brier_score_loss as _fast_brier, fast_log_loss as _fast_ll


def brier(yy: np.ndarray, pp: np.ndarray) -> float:
    """Bootstrap-resample Brier score via the numba-fast kernel."""
    return float(_fast_brier(yy, pp))


def log_loss(yy: np.ndarray, pp: np.ndarray) -> float:
    """Bootstrap-resample log-loss via the numba-fast kernel."""
    return float(_fast_ll(yy, pp))


def ll_per_row(yy: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Per-row cross-entropy contribution, used by the BCa jackknife's O(n) exact-algebraic
    leave-one-out formula for log-loss."""
    _eps = np.finfo(np.asarray(pp).dtype).eps
    _pc = np.clip(pp, _eps, 1.0 - _eps)
    return np.where(np.asarray(yy) == 1, -np.log(_pc), -np.log(1.0 - _pc))


def brier_per_row(yy: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Per-row squared-error contribution, used by the BCa jackknife's O(n) exact-algebraic
    leave-one-out formula for Brier score."""
    _d = np.asarray(pp, dtype=np.float64) - np.asarray(yy, dtype=np.float64)
    return _d * _d
