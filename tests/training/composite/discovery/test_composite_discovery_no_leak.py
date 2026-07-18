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
    filtered_train_df = pd.DataFrame(
        {
            "x0": np.arange(n, dtype=np.float64),
            "x1": np.arange(n, dtype=np.float64) * 0.5,
        }
    )
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
    assert (
        list(filtered_train_df.columns) == cols_before
    ), f"S04: filtered_train_df.columns mutated: before={cols_before} after={list(filtered_train_df.columns)}"


def test_composite_discovery_disc_df_build_does_not_leak_mutation_to_caller():
    """The discovery helper MUST NOT let a write to its returned frame mutate
    the caller's ``filtered_train_df``.

    The real invariant is MUTATION ISOLATION, not physical buffer identity.
    Under pandas Copy-on-Write (default in pandas 3.0 / opt-in in 2.x, active
    on the prod box 2026-05-27) ``.assign(...)`` returns a frame whose
    unchanged columns LAZILY SHARE buffers with the source -- ``np.shares_memory``
    reports True -- yet CoW guarantees any subsequent write triggers a copy, so
    no mutation ever leaks. A physical ``shares_memory`` assertion therefore
    fails-but-safe under CoW; assert the behaviour (write isolation) instead, so
    the test holds whether CoW shares buffers or not. The bad pattern this guards
    against (``copy(deep=False) + setitem`` promoting a shared block) WOULD leak a
    value mutation and is caught here.
    """
    n = 200
    filtered_train_df = pd.DataFrame(
        {
            "x0": np.full(n, 123.456, dtype=np.float64),
            "x1": np.full(n, 7.89, dtype=np.float64),
        }
    )
    y = np.arange(n, dtype=np.float64)
    _disc = _build_disc_df_for_target(filtered_train_df, "target_A", y)
    caller_x0_before = filtered_train_df["x0"].to_numpy().copy()
    caller_x1_before = filtered_train_df["x1"].to_numpy().copy()
    # Write through the discovery frame's feature columns. If the helper shared
    # a writable block with the caller (copy(deep=False)+setitem), this poke
    # would bleed back; ``.assign`` + CoW isolates it.
    _disc.loc[:, "x0"] = -1.0
    _disc.loc[:, "x1"] = -2.0
    np.testing.assert_array_equal(
        filtered_train_df["x0"].to_numpy(),
        caller_x0_before,
        err_msg="S04: write to _disc['x0'] leaked into caller's filtered_train_df",
    )
    np.testing.assert_array_equal(
        filtered_train_df["x1"].to_numpy(),
        caller_x1_before,
        err_msg="S04: write to _disc['x1'] leaked into caller's filtered_train_df",
    )
