"""Attaching the target for discovery must not copy the train frame, nor touch the caller's frame.

Under pandas 1.5-2.x a list-column selection plus ``concat``'s default ``copy=True`` materialised up to two transient
copies of the whole train frame per regression target just to attach y. Passing ``copy=False`` did not fix it: on 2.x
with copy-on-write off, which is the default and what this project runs, ``concat`` copies the block regardless. The
new frame is built from the caller's own Series instead, which shares them; pandas 3 is zero-copy either way and the
same assertions hold there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_discovery_helpers import _build_disc_df_for_target


def _train_frame(n: int = 1000) -> pd.DataFrame:
    """A float train frame without the target column, as the per-target loop receives it."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(6)})


def test_feature_columns_share_memory_with_the_train_frame():
    """The discovery frame reads the caller's feature data in place instead of copying it."""
    train = _train_frame()
    out = _build_disc_df_for_target(train, "y", np.arange(len(train), dtype=np.float64))
    assert np.shares_memory(out["f0"].to_numpy(), train["f0"].to_numpy()), "the feature data was copied"


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
