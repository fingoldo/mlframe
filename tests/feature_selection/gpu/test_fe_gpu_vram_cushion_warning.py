"""A GPU dispatch refused for lack of free VRAM is announced at WARNING (with the numbers), a granted one is not, and the answer itself is unchanged."""

from __future__ import annotations

import logging
import threading

import pytest

from mlframe.feature_selection.filters import _fe_gpu_vram as v
from mlframe.utils.log_throttle import reset_throttle_counts

_MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget the per-process warning count so each test sees its own first warning.

    Also releases our cupy default pool's cached-free blocks: ``fe_gpu_has_vram_cushion``'s caller-supplied-probe
    path is NOT a pure function of its ``free_b``/``total_b`` arguments alone - when the primary check fails it
    consults ``_own_pool_free_bytes()`` (the REAL process-wide pool) as a rescue (see its own code comment: a
    caller's ``memGetInfo`` undercounts blocks our pool holds cached-free). An earlier GPU test in the same
    process (e.g. a real device_sort_safe/batch_pair_mi allocation) can leave that real pool holding enough
    cached-free bytes to rescue this file's SYNTHETIC near-full-card scenarios, flipping an expected decline to
    a grant - this is what made the suite flaky only inside a full-directory run. Releasing the pool here makes
    the synthetic scenarios deterministic regardless of run order / prior GPU activity in this process."""
    reset_throttle_counts()
    v._release_own_pool()
    yield
    reset_throttle_counts()


def _declined(caplog) -> list:
    """The refusal messages logged by THIS thread (a worker thread left over from an earlier test of a full run can refuse and log at the same moment)."""
    me = threading.get_ident()
    return [r.getMessage() for r in caplog.records if "GPU path declined" in r.getMessage() and r.thread == me]


def test_a_refusal_warns_with_the_numbers_and_still_returns_false(caplog):
    """1.5 GiB free of 4 GiB with 1 GiB needed leaves 0.5 GiB against a 1 GiB cushion: refused, and the warning says how much was free, needed and required."""
    with caplog.at_level(logging.WARNING, logger=v.logger.name):
        ok = v.fe_gpu_has_vram_cushion(1024 * _MIB, free_b=1536 * _MIB, total_b=4096 * _MIB)
    assert ok is False
    msgs = _declined(caplog)
    assert len(msgs) == 1
    assert "1536 MiB free" in msgs[0] and "1024 MiB needed" in msgs[0] and "1024 MiB cushion" in msgs[0] and "4096 MiB" in msgs[0]


def test_a_granted_request_is_silent(caplog):
    """3 GiB free of 4 GiB with 1 GiB needed keeps the cushion: granted, nothing logged about a refusal."""
    with caplog.at_level(logging.WARNING, logger=v.logger.name):
        ok = v.fe_gpu_has_vram_cushion(1024 * _MIB, free_b=3072 * _MIB, total_b=4096 * _MIB)
    assert ok is True
    assert not _declined(caplog)


def test_the_warning_is_limited_per_process(caplog):
    """Many refusals in a row produce a handful of warnings, not one per dispatch."""
    with caplog.at_level(logging.WARNING, logger=v.logger.name):
        for _ in range(50):
            assert v.fe_gpu_has_vram_cushion(1024 * _MIB, free_b=1024 * _MIB, total_b=4096 * _MIB) is False
    assert 1 <= len(_declined(caplog)) <= 6
