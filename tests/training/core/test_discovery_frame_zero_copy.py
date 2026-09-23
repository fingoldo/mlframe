"""The per-target discovery frame borrows the train frame's columns instead of copying them (INT-12, PRF-15).

Discovery runs once per regression target, so a frame copy there is paid K times over the whole train frame - the one
thing the pipeline must never do on a 100 GB input. The frame still has to be a new object (the caller's must not gain
the injected target column), which is exactly the pair of properties asserted here: new frame, shared columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_discovery_helpers import _build_disc_df_for_target


def _train_frame(n: int = 500) -> pd.DataFrame:
    """A small float train frame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})


def test_the_discovery_frame_shares_the_feature_columns_with_the_train_frame():
    """Every feature column of the discovery frame is the train frame's buffer, on any supported pandas."""
    df = _train_frame()
    out = _build_disc_df_for_target(df, "target", np.ones(len(df)))
    for col in df.columns:
        assert np.shares_memory(out[col].to_numpy(), df[col].to_numpy()), f"column '{col}' was copied (pandas {pd.__version__})"


def test_the_train_frame_does_not_gain_the_injected_target():
    """The frame is new: the injected column and its values stay out of the caller's frame."""
    df = _train_frame()
    before = list(df.columns)
    out = _build_disc_df_for_target(df, "target", np.full(len(df), 7.0))
    assert list(df.columns) == before
    np.testing.assert_allclose(out["target"].to_numpy(), 7.0)


def test_an_existing_target_column_is_replaced_only_in_the_discovery_frame():
    """When the target name is already a column, the discovery frame carries the aligned y and the caller keeps its own."""
    df = _train_frame()
    df["target"] = 0.0
    out = _build_disc_df_for_target(df, "target", np.full(len(df), 3.0))
    np.testing.assert_allclose(out["target"].to_numpy(), 3.0)
    np.testing.assert_allclose(df["target"].to_numpy(), 0.0)
    assert np.shares_memory(out["f1"].to_numpy(), df["f1"].to_numpy())
