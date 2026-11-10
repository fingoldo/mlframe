"""The dominant-pair scan in operand tiles picks the pair one call over all pairs picks, and a starved budget does not turn it into one call per pair."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_rank as sp
from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as bp


def _setup(n_ops=14, n=1500, seed=0):
    """Operands (one with a planted relation to y), the continuous target and the pair keys in scan order."""
    r = np.random.default_rng(seed)
    ops = [r.standard_normal(n).astype(np.float32) for _ in range(n_ops)]
    y = (ops[3] * ops[9] + 0.2 * r.standard_normal(n)).astype(np.float32)
    pairs = [(i, j) for i in range(n_ops) for j in range(i + 1, n_ops)]
    return ops, y, pairs


def test_tiled_scan_equals_one_call_over_all_pairs_with_few_calls(monkeypatch):
    """With a budget of four operand rows the winner is (3, 9), as with one call, and the dispatcher is called per tile group, not per pair."""
    ops, y, pairs = _setup()
    cached = lambda i: ops[i]
    monkeypatch.setattr(sp, "_prescan_budget_bytes", lambda n_rows, dtype: (4 * n_rows * np.dtype(dtype).itemsize, n_rows * np.dtype(dtype).itemsize))
    calls = []
    real = bp.dispatch_batch_pair_usability_corr

    def counting(*a, **k):
        """Counts dispatcher calls."""
        calls.append(len(a[3]))
        return real(*a, **k)

    monkeypatch.setattr(bp, "dispatch_batch_pair_usability_corr", counting)
    got = sp._dominant_pair_by_linear_corr(pairs, y, cached)
    assert got == (3, 9)
    assert sum(calls) == len(pairs)
    assert len(calls) <= 16  # 14 operands in chunks of two: 7 chunks, at most 28 groups, far below 91 pairs


def test_tie_goes_to_the_earliest_pair(monkeypatch):
    """Identical operands give equal scores for every pair; the first pair in scan order wins."""
    n = 800
    r = np.random.default_rng(1)
    base = r.standard_normal(n).astype(np.float32)
    ops = [base.copy() for _ in range(6)]
    y = (base + 0.1 * r.standard_normal(n)).astype(np.float32)
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    monkeypatch.setattr(sp, "_prescan_budget_bytes", lambda n_rows, dtype: (2 * n_rows * np.dtype(dtype).itemsize, n_rows * np.dtype(dtype).itemsize))
    assert sp._dominant_pair_by_linear_corr(pairs, y, lambda i: ops[i]) == pairs[0]
