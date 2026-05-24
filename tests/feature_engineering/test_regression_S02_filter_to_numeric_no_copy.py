"""Regression sensor for S02: ``_filter_to_numeric`` must NOT call ``.copy()`` on the full frame when bool columns are present.

The CRITICAL "no df.copy() on hot paths" rule (CLAUDE.md "Memory / RAM constraints") forbids cloning a 100+ GB pandas frame to satisfy a bool->int8 promotion. The pre-fix code in ``mlframe.training._pipeline_extensions.apply_preprocessing_extensions._filter_to_numeric`` did ``_df = _df.copy()`` whenever any bool column was present, doubling peak RAM on a wide frame.

Post-fix: the function casts bool columns in place per-column on the caller's frame (single-column ``_df[c] = _df[c].astype(np.int8)`` is a block-level dtype swap that does NOT clone the float / numeric blocks). The caller's frame DOES see the bool->int8 promotion -- this is the documented price of obeying the no-full-frame-copy rule on 100+GB workloads. The float blocks keep their original buffers (verified via ``np.shares_memory``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training._pipeline_extensions import _filter_to_numeric


def _make_frame_with_bools(n: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x0": rng.standard_normal(n).astype(np.float32),
            "x1": rng.standard_normal(n).astype(np.float32),
            "is_active": rng.integers(0, 2, n).astype(bool),
            "is_after_ps": rng.integers(0, 2, n).astype(bool),
        }
    )


def test_filter_to_numeric_mutates_caller_frame_bools_in_place() -> None:
    """The no-full-frame-copy contract: after ``_filter_to_numeric``, the caller's frame must have its bool columns promoted to int8 in place (proving the function did NOT do its work on a sandbox clone). Pre-fix ``_df = _df.copy()`` worked on a clone, so the caller's bool columns were left as bool dtype."""
    df = _make_frame_with_bools(10_000)
    assert df["is_active"].dtype == np.bool_
    assert df["is_after_ps"].dtype == np.bool_

    _ = _filter_to_numeric(df)

    # Post-fix: caller's frame has int8 dtype for the previously-bool columns. Pre-fix: still bool (the cast lived on the copy that was thrown away).
    assert df["is_active"].dtype == np.int8, (
        f"caller's 'is_active' column dtype is {df['is_active'].dtype!r}; expected int8. "
        "_filter_to_numeric copied the frame (violating the 100GB no-copy rule) and only "
        "mutated the throwaway clone."
    )
    assert df["is_after_ps"].dtype == np.int8


def test_filter_to_numeric_float_buffers_survive_bool_promotion() -> None:
    """Stronger no-copy proof: the unchanged float-block buffers must STILL alias the caller's frame after the function returns. Per-column in-place dtype mutation on the bool columns must not perturb the float block. Pre-fix ``_df = _df.copy()`` consolidated all blocks into a fresh BlockManager -- the float pointers were severed before the cast even ran."""
    df = _make_frame_with_bools(10_000)
    x0_buf_before = df["x0"].values
    x1_buf_before = df["x1"].values

    _ = _filter_to_numeric(df)

    # The caller's float columns must still point at the original buffers. ``shares_memory`` is stricter than ``may_share_memory`` and catches both partial overlaps and full aliasing -- exactly what we need to verify the function did not clone the whole BlockManager.
    assert np.shares_memory(df["x0"].values, x0_buf_before), (
        "caller's 'x0' float buffer differs from pre-call buffer -- _filter_to_numeric "
        "rebuilt the BlockManager (consistent with a full ``.copy()`` violating the 100GB no-copy rule)."
    )
    assert np.shares_memory(df["x1"].values, x1_buf_before)


def test_filter_to_numeric_returns_filtered_subset_and_drops_object() -> None:
    """Behavioural sanity: object columns ARE dropped, bool columns ARE promoted, numeric columns survive."""
    rng = np.random.default_rng(3)
    n = 1000
    df = pd.DataFrame(
        {
            "f0": rng.standard_normal(n).astype(np.float64),
            "flag": rng.integers(0, 2, n).astype(bool),
            "label": np.array([f"L{i % 5}" for i in range(n)], dtype=object),
        }
    )
    out, dropped = _filter_to_numeric(df)
    assert isinstance(out, pd.DataFrame)
    assert dropped == ["label"], f"only 'label' object column should be dropped; got dropped={dropped!r}"
    assert "flag" in out.columns
    assert out["flag"].dtype == np.int8
