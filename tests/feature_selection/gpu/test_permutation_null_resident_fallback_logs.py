"""Every fallback of the resident maxT floors to the CPU kernels logs a warning instead of a debug line."""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from mlframe.feature_selection.filters import _permutation_null_pair_resident as pr
from mlframe.feature_selection.filters import _permutation_null_resident as r1
from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _no_cupy(monkeypatch):
    """Make ``import cupy`` fail and clear the log throttle and the circuit breakers around each test."""
    monkeypatch.setitem(sys.modules, "cupy", None)
    reset_throttle_counts()
    yield
    pr.reset_pair_maxt_gpu_circuit_breaker()
    r1.reset_order1_maxt_gpu_circuit_breaker()


def test_pair_floor_fault_is_logged_at_warning(caplog):
    """A faulted order-2 resident floor returns None, trips the breaker and warns."""
    n = 64
    data = (np.arange(n * 3).reshape(n, 3) % 3).astype(np.int32)
    y = np.arange(n) % 2
    with caplog.at_level(logging.WARNING):
        out = pr.pooled_pair_permutation_null_joint_mi_floor_cupy(
            data, np.array([0, 1]), np.array([1, 2]), np.array([3, 3, 3]), y, np.array([0.5, 0.5]), n_permutations=4, random_seed=1
        )
    assert out is None
    assert any(rec.levelno >= logging.WARNING and "pair-maxT floor faulted" in rec.getMessage() for rec in caplog.records)


def test_order1_shuffle_gen_fault_is_logged_at_warning(caplog):
    """A faulted device shuffle-gen returns None and warns."""
    with caplog.at_level(logging.WARNING):
        out = r1.gen_target_shuffles_cupy(np.arange(32) % 2, 4, np.int32, 3)
    assert out is None
    assert any(rec.levelno >= logging.WARNING and "shuffle-gen failed" in rec.getMessage() for rec in caplog.records)


def test_pair_floor_vram_decline_is_logged(monkeypatch, caplog):
    """A VRAM-cushion refusal that sends the order-2 floor to the CPU says so."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda nbytes: False)
    with caplog.at_level(logging.WARNING):
        assert pr.pair_maxt_perm_null_gpu_enabled(1000, 50) is False
    assert any(rec.levelno >= logging.WARNING and "no VRAM cushion" in rec.getMessage() for rec in caplog.records)
