"""Regression sensor for S04: ``_phase_composite_discovery`` per-target loop
does ``_disc_df = filtered_train_df.copy(deep=False); _disc_df[_tname_disc] = _y_train_aligned``.
Pandas ``copy(deep=False)`` shares the BlockManager with the caller's frame, so the
subsequent ``[_tname_disc] = arr`` setitem can mutate the SHARED block depending on
dtype + block layout - the target column intermittently appears on
``filtered_train_df`` post-loop and feeds back into subsequent training iterations
as a feature.

The fix is to construct ``_disc_df`` via ``filtered_train_df.assign(**{_tname_disc: arr})``
(which always builds a fresh BlockManager) instead of ``.copy(deep=False)`` + setitem.

This sensor pins the safe behaviour directly on the production helper that
performs the per-target discovery dataframe build, so it FAILS on the
``copy(deep=False)+setitem`` pattern and PASSES on the ``.assign`` pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_discovery import (
    _build_disc_df_for_target,
)


def test_composite_discovery_disc_df_build_does_not_leak_target_into_caller_pd():
    """Build _disc_df for target_A, then for target_B. After both calls,
    the caller's filtered_train_df.columns MUST contain ONLY the original
    feature columns - neither target_A nor target_B should leak in.
    """
    n = 200
    filtered_train_df = pd.DataFrame({
        "x0": np.arange(n, dtype=np.float64),
        "x1": np.arange(n, dtype=np.float64) * 0.5,
    })
    cols_before = list(filtered_train_df.columns)

    y_a = np.linspace(0.0, 1.0, n).astype(np.float64)
    y_b = (np.linspace(0.0, 1.0, n) ** 2).astype(np.float64)

    _disc_a = _build_disc_df_for_target(filtered_train_df, "target_A", y_a)
    _disc_b = _build_disc_df_for_target(filtered_train_df, "target_B", y_b)

    assert "target_A" in _disc_a.columns
    assert "target_B" in _disc_b.columns
    assert "target_A" not in filtered_train_df.columns, (
        "S04: target_A leaked back into caller's filtered_train_df after "
        "_build_disc_df_for_target. The discovery helper must not mutate "
        f"the caller's frame. Columns now: {list(filtered_train_df.columns)}"
    )
    assert "target_B" not in filtered_train_df.columns, (
        "S04: target_B leaked back into caller's filtered_train_df after "
        "_build_disc_df_for_target. Per-target loop accumulates leakage. "
        f"Columns now: {list(filtered_train_df.columns)}"
    )
    assert list(filtered_train_df.columns) == cols_before, (
        f"S04: filtered_train_df.columns mutated: before={cols_before} "
        f"after={list(filtered_train_df.columns)}"
    )


def test_composite_discovery_disc_df_build_does_not_share_block_with_caller():
    """The discovery helper MUST return a frame whose underlying block buffers
    do NOT alias the caller's frame. ``copy(deep=False) + setitem`` shares the
    BlockManager and exposes the caller to silent mutation through any future
    pandas-internal setitem path that promotes a shared block (e.g. dtype-
    matching multi-column writes); ``.assign(...)`` always constructs a fresh
    BlockManager so the invariant holds across pandas versions.

    The aliasing check via ``np.shares_memory`` catches the bad pattern
    deterministically, even on pandas versions where the value-level leak
    only fires under specific dtype + block layouts.
    """
    n = 200
    filtered_train_df = pd.DataFrame({
        "x0": np.full(n, 123.456, dtype=np.float64),
        "x1": np.full(n, 7.89, dtype=np.float64),
    })
    y = np.arange(n, dtype=np.float64)
    _disc = _build_disc_df_for_target(filtered_train_df, "target_A", y)
    # Even if pandas' CoW happens to spare us today on a value-level leak, the
    # ``copy(deep=False)`` pattern leaves the discovery frame's float64 column
    # buffer aliased to the caller's frame. ``.assign(...)`` builds a fresh
    # BlockManager that does NOT alias.
    for col in ("x0", "x1"):
        assert not np.shares_memory(
            filtered_train_df[col].to_numpy(), _disc[col].to_numpy()
        ), (
            f"S04: column {col!r} in _disc_df aliases the caller's "
            f"filtered_train_df. The helper must use ``.assign(...)`` "
            f"or a fresh-BlockManager constructor (``copy(deep=False)`` "
            f"shares blocks)."
        )
