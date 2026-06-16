"""Regression: ``cheap_row_argmax_scan`` must select the SAME budget-truncated set
of column triples regardless of input column order.

Bug (fixed): the scan enumerated ``combinations(cols, 3)`` in raw input-column order
and truncated at ``max_triples``. Because C(p, 3) far exceeds the budget for any
realistic p, reversing the input columns fed a DIFFERENT first-``budget`` set of
triples into the candidate pool, seeded different row-argmax engineered features, and
changed the downstream MRMR greedy selection -- breaking the column-order-invariance
contract (TestColumnOrderInvariance[MRMR]). The fix enumerates triples over the
NAME-sorted column order so the budgeted set is invariant under any column permutation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._conditional_gate_fe import cheap_row_argmax_scan


def _frame(n=400, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(p)})
    # a genuine argmax-coded target so some triples actually clear the margin
    code = np.argmax(np.stack([X["f0"], X["f1"], X["f2"]], axis=1), axis=1)
    y = (code >= 1).astype(np.int64)
    return X, y


def test_budgeted_triple_set_is_column_order_invariant():
    # p=10 -> C(10,3)=120 triples >> budget 40, so the budget genuinely truncates
    # and an order-sensitive enumeration would pick a different set on reversal.
    X, y = _frame(p=10)
    rev = list(X.columns)[::-1]
    hits_fwd = cheap_row_argmax_scan(X, y, max_triples=40)
    hits_rev = cheap_row_argmax_scan(X[rev], y, max_triples=40)

    # Compare the SET of scanned triples (order-insensitive within each triple, since
    # argmax(a,b,c) over a sorted triple is the canonical representative).
    set_fwd = {tuple(sorted(h.cols)) for h in hits_fwd}
    set_rev = {tuple(sorted(h.cols)) for h in hits_rev}
    assert set_fwd == set_rev, (
        f"row-argmax budgeted triple set is column-order dependent: "
        f"fwd-only={set_fwd - set_rev} rev-only={set_rev - set_fwd}"
    )
    # The budget actually bit (otherwise the test would not discriminate the bug).
    assert len(hits_fwd) == 40
