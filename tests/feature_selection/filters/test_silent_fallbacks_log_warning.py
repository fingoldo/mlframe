"""A fallback that substitutes a decision after a failed probe must say so at WARNING, with the exception that caused it.

These gates run without a GPU in the test environment, so the failure is injected by replacing the probe with one that raises.
"""

from __future__ import annotations

import logging

import pytest

import numpy as np

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as pair_usability
from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_rank
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi
from mlframe.feature_selection.filters._pair_operand_blocks import pair_chunks_by_operand_block
from mlframe.utils.log_throttle import reset_throttle_counts


class _RaisingChooser:
    """Stands in for a kernel-tuning spec whose ``choose`` call fails."""

    def choose(self, **_kwargs):
        """Raise the failure the tuner would hit when its cache is unreadable."""
        raise RuntimeError("tuning cache unreadable")


@pytest.fixture(autouse=True)
def _fresh_throttle_counts():
    """Forget the throttle counts of these keys so each test sees its first warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_basis_build_choose_failure_warns_and_keeps_single_kernel(monkeypatch, caplog):
    """A failing tuner lookup keeps the single-kernel basis build and says why at WARNING."""
    import mlframe.feature_selection.filters.hermite_fe._basis_fused_gpu as basis_fused

    monkeypatch.setattr(basis_fused, "_SPEC", _RaisingChooser())
    with caplog.at_level(logging.WARNING, logger=basis_fused.logger.name):
        assert basis_fused.basis_fused_chosen(1000) is True
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("KTC choose() failed" in m and "RuntimeError" in m and "tuning cache unreadable" in m for m in messages)


def test_pair_usability_choose_failure_warns_and_keeps_device(monkeypatch, caplog):
    """A failing tuner lookup keeps the device backend for the pair usability batch and says why at WARNING."""
    monkeypatch.setattr(pair_usability, "_SPEC", _RaisingChooser())
    with caplog.at_level(logging.WARNING, logger=pair_usability.logger.name):
        assert pair_usability._device_wins(5000) is True
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("KTC choose() failed" in m and "RuntimeError" in m for m in messages)


def test_unreadable_host_memory_warns_and_budgets_zero(monkeypatch, caplog):
    """Unreadable host memory budgets zero bytes and names the exception at WARNING, so the smallest path is visibly chosen."""

    def _unreadable():
        """Simulate a host-memory probe the OS refuses to answer."""
        raise OSError("meminfo unavailable")

    monkeypatch.setattr(_streamed_gpu_mi, "_host_free_bytes", _unreadable)
    with caplog.at_level(logging.WARNING, logger=_step_pairs_rank.logger.name):
        assert _step_pairs_rank._host_free_bytes_safe() == 0
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("host free memory unreadable" in m and "OSError" in m for m in messages)


def test_column_block_width_honours_monkeypatched_override(monkeypatch):
    """A block width set on the module overrides the VRAM-derived width, which is what the env override feeds."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_batch_gpu as fourier_batch

    monkeypatch.setattr(fourier_batch, "_BLOCK_COLS", 3)
    assert fourier_batch._column_block_width(10, 50) == 3


def test_pair_chunks_by_operand_block_keeps_operands_within_limit():
    """Every chunk touches at most the allowed number of operand columns, and every pair lands in exactly one chunk."""
    pair_a = np.array([0, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    pair_b = np.array([1, 2, 2, 3, 4, 5, 6], dtype=np.int64)
    out = pair_chunks_by_operand_block(pair_a, pair_b, max_operands=4)
    seen_pairs = sorted(int(i) for c in out for i in c)
    assert seen_pairs == list(range(len(pair_a)))
    for c in out:
        operands = {int(pair_a[i]) for i in c} | {int(pair_b[i]) for i in c}
        assert len(operands) <= 4
