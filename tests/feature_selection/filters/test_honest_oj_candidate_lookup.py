"""The fused floor keeps every pool column's per-permutation MI; a candidate's exceedance count is read from it only for the same stream and rows."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters import _honest_oj_fused as F


def _hook(n_out=3, n_perm=50, seed=7, n=100, rows=None):
    """A codes hook whose binner holds a kept null for pool columns 0, 1 and 4 (column 4 is an engineered one past the binner's outputs)."""
    rows = np.arange(n) if rows is None else rows
    mi = np.random.default_rng(0).random((n_perm, 3))
    binner = SimpleNamespace(_out_names=[f"c{i}" for i in range(n_out)], _fused_null_mi=(F._null_key(seed, n_perm, n, rows), {0: 0, 1: 1, 4: 2}, mi))
    return SimpleNamespace(binner=binner), mi


def test_count_is_read_from_the_head_of_the_stream():
    """A shorter run of the same stream counts over the first permutations of the kept matrix."""
    hook, mi = _hook()
    rows = np.arange(100)
    assert F.fused_column_nfailed(hook, 1, 0.5, 20, 7, 100, rows) == int((mi[:20, 1] >= 0.5).sum())
    assert F.fused_column_nfailed(hook, 0, 0.5, 50, 7, 100, rows) == int((mi[:, 0] >= 0.5).sum())


def test_other_stream_rows_or_columns_are_not_served():
    """Another seed, more permutations than kept, other rows, a column outside the pool or past the binner's outputs: ``None``."""
    hook, _ = _hook()
    rows = np.arange(100)
    assert F.fused_column_nfailed(hook, 1, 0.5, 20, 8, 100, rows) is None
    assert F.fused_column_nfailed(hook, 1, 0.5, 51, 7, 100, rows) is None
    assert F.fused_column_nfailed(hook, 1, 0.5, 20, 7, 100, np.arange(50)) is None
    assert F.fused_column_nfailed(hook, 2, 0.5, 20, 7, 100, rows) is None
    assert F.fused_column_nfailed(hook, 4, 0.5, 20, 7, 100, rows) is None
    assert F.fused_column_nfailed(SimpleNamespace(binner=None), 1, 0.5, 20, 7, 100, rows) is None
