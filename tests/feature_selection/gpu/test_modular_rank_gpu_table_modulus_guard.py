"""A modulus above the int8 residue-table range sends the GPU modular ranking to the CPU kernel with a logged warning, instead of wrapping residues negative in shared memory."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from mlframe.feature_selection.filters import _pairwise_modular_rank_gpu as rg
from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fake_cupy(monkeypatch):
    """A cupy stand-in so the guard branches run on a CPU-only host, and a clean log throttle."""
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.delenv("MLFRAME_MODULAR_RANK_KERNEL", raising=False)
    reset_throttle_counts()


def _inputs(mods):
    """A tiny pair-ranking problem over the given moduli."""
    rng = np.random.default_rng(0)
    cols = rng.integers(0, 50, size=(4, 200)).astype(np.int64)
    ia = np.array([0, 1], dtype=np.int64)
    ib = np.array([2, 3], dtype=np.int64)
    ycls = rng.integers(0, 2, size=200).astype(np.int64)
    return cols, ia, ib, np.asarray(mods, dtype=np.int64), ycls


def test_table_kernel_refuses_a_modulus_above_the_int8_range(caplog):
    """The default table kernel stores residues as signed bytes, so a modulus of 200 returns None and says why."""
    cols, ia, ib, mods, ycls = _inputs([2, 3, 200])
    with caplog.at_level(logging.WARNING):
        out = rg.residue_rank_scores_gpu(cols, ia, ib, None, mods, ycls, 2)
    assert out is None
    assert any("int8 residue table" in r.getMessage() for r in caplog.records)


def test_shared_memory_overflow_is_logged(caplog, monkeypatch):
    """A count matrix over the shared-memory limit falls back to the CPU kernel with a warning instead of silently."""
    monkeypatch.setenv("MLFRAME_MODULAR_RANK_KERNEL", "4")
    cols, ia, ib, mods, ycls = _inputs([2, 3, 6000])
    with caplog.at_level(logging.WARNING):
        out = rg.residue_rank_scores_gpu(cols, ia, ib, None, mods, ycls, 2)
    assert out is None
    assert any("do not fit" in r.getMessage() for r in caplog.records)
