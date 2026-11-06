"""The row budget of the pair-residual check is the same whatever backend scores it.

The GPU backend used a budget 25 times larger, so the same frame could be scored on the device and stop the FE search, and fall below the row
minimum on the CPU and let it run. The host's tuning cache chose the backend (audit PARG2-1, wave 3, High).
"""

from __future__ import annotations

from mlframe.feature_selection.filters import _fe_sufficient_summary_pairs as fsp


def test_gpu_and_cpu_budgets_are_the_same():
    """There is one budget for the pair-residual check; the GPU name is an alias of it."""
    assert fsp.PAIR_WORK_BUDGET_GPU == fsp.PAIR_WORK_BUDGET
