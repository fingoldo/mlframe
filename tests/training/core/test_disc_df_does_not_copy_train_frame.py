"""Attaching the target for discovery must not copy the train frame, nor touch the caller's frame.

Under pandas 1.5-2.x a list-column selection plus ``concat``'s default ``copy=True`` materialised up to two transient
copies of the whole train frame per regression target just to attach y. Passing ``copy=False`` did not fix it: on 2.x
with copy-on-write off, which is the default and what this project runs, ``concat`` copies the block regardless. The
copy is kept on purpose: sharing the buffers without CoW would let a write to the discovery frame land in the train
frame. What is pinned here is that isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_discovery_helpers import _build_disc_df_for_target


def _train_frame(n: int = 1000) -> pd.DataFrame:
    """A float train frame without the target column, as the per-target loop receives it."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(6)})


def test_a_write_to_the_discovery_frame_never_reaches_the_train_frame():
    """Mutation isolation is the contract, not buffer identity.

    This used to assert ``np.shares_memory`` on a feature column. On pandas 2.x with copy-on-write off that cannot
    coexist with isolation: a frame sharing the caller's buffers passes a write straight through to the caller's
    frame, which the per-target loop would then read as a feature. So the copy stays, and what is pinned is the
    property that matters.
    """
    train = _train_frame()
    before = train["f0"].to_numpy().copy()
    out = _build_disc_df_for_target(train, "y", np.arange(len(train), dtype=np.float64))
    out["f0"] = out["f0"] + 1.0
    out.loc[out.index[0], "f1"] = 1e9
    np.testing.assert_array_equal(train["f0"].to_numpy(), before, err_msg="a write to the discovery frame leaked into the train frame")
    assert train.loc[train.index[0], "f1"] != 1e9, "an in-place cell write leaked into the train frame"


def test_the_callers_frame_is_left_without_the_target():
    """Injecting y must never leak a target column into the caller's frame (the per-target leakage the helper prevents)."""
    train = _train_frame()
    before = list(train.columns)
    out = _build_disc_df_for_target(train, "y", np.arange(len(train), dtype=np.float64))
    assert list(train.columns) == before
    assert "y" in out.columns and list(out.columns)[:-1] == before


def test_an_existing_target_column_is_replaced_not_duplicated():
    """When the frame already carries the target, the injected values replace it and the column appears once."""
    train = _train_frame()
    train["y"] = -1.0
    y = np.arange(len(train), dtype=np.float64)
    out = _build_disc_df_for_target(train, "y", y)
    assert list(out.columns).count("y") == 1
    np.testing.assert_array_equal(out["y"].to_numpy(), y)
    assert (train["y"] == -1.0).all(), "the caller's own target column must be untouched"
