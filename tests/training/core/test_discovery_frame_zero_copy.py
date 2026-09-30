"""The per-target discovery frame is a new frame that carries the train frame's feature values and never writes through to it.

Buffer sharing is deliberately not asserted: on pandas 2.x with copy-on-write off (the default, and what this project runs)
a frame sharing the caller's buffers passes a write to the discovery frame straight into the train frame, which the
per-target loop would then read as a feature. ``test_disc_df_does_not_copy_train_frame.py`` pins that isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_discovery_helpers import _build_disc_df_for_target


def _train_frame(n: int = 500) -> pd.DataFrame:
    """A small float train frame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})


def test_the_discovery_frame_carries_the_feature_columns_of_the_train_frame():
    """Every feature column of the discovery frame holds the train frame's values, in the same order."""
    df = _train_frame()
    out = _build_disc_df_for_target(df, "target", np.ones(len(df)))
    assert list(df.columns), "the fixture frame has no columns to compare"
    assert list(out.columns) == [*df.columns, "target"]
    for col in df.columns:
        np.testing.assert_array_equal(out[col].to_numpy(), df[col].to_numpy())


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
    np.testing.assert_array_equal(out["f1"].to_numpy(), df["f1"].to_numpy())
