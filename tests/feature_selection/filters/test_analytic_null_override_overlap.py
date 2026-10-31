"""Overlapping ``analytic_null_min_n_lowered`` blocks do not leak or lose their floor (CON2-2).

The override used to be a process-wide global that each block saved and restored. When blocks overlapped across threads, a block
that exited first restored a stale value, so a later block could leave the lowered floor set for the rest of the process, or drop
the floor of a block that was still running. Each block now owns one entry in a locked list.
"""

import pytest

from mlframe.feature_selection.filters import _analytic_mi_null as amn


@pytest.fixture(autouse=True)
def _default_threshold(monkeypatch):
    """Use the built-in threshold so the floors under test are the only thing that moves it."""
    monkeypatch.delenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", raising=False)


def test_overlapping_blocks_keep_the_floor_until_the_last_one_exits():
    """Exiting the first of two overlapping blocks leaves the second block's floor in force."""
    first = amn.analytic_null_min_n_lowered(20_000)
    second = amn.analytic_null_min_n_lowered(15_000)
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    assert amn.analytic_null_min_n() == 15_000, "the still-running block lost its lowered floor"
    second.__exit__(None, None, None)


def test_no_override_survives_when_every_block_has_exited():
    """After all overlapping blocks exit, the threshold is back at the default, not stuck at an earlier block's floor."""
    first = amn.analytic_null_min_n_lowered(20_000)
    second = amn.analytic_null_min_n_lowered(15_000)
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    second.__exit__(None, None, None)
    assert amn.analytic_null_min_n() == amn._ANALYTIC_NULL_MIN_N_DEFAULT
