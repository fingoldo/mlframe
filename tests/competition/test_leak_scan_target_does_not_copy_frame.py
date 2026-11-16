"""With a target, the density leak scan stacks the target column in front of the frame's values instead of copying the frame."""

import warnings

import numpy as np
import pandas as pd

from mlframe.competition.leak_scan import sort_by_density_leak_scan


def _frame_and_target():
    """A small integer-valued frame with a binary target."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 3, (40, 5)).astype(float), columns=list("abcde"))
    return df, rng.integers(0, 2, 40).astype(float)


def test_target_path_matches_a_frame_that_already_holds_the_target(monkeypatch):
    """The result equals the scan of a frame that already has the target column, and no DataFrame.copy is made."""
    df, target = _frame_and_target()
    expected_df = df.copy()
    expected_df.insert(0, "__target__", target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = sort_by_density_leak_scan(expected_df, target=None)

        def no_copy(self, *a, **k):
            """Fail if the frame is copied on the target path."""
            raise AssertionError("the target path must not copy the frame")

        monkeypatch.setattr(pd.DataFrame, "copy", no_copy)
        got = sort_by_density_leak_scan(df, target=target)
    for key in ("row_order", "col_order", "row_density", "col_density"):
        np.testing.assert_array_equal(got[key], expected[key])
    np.testing.assert_allclose(got["overlap_scores"], expected["overlap_scores"], rtol=0, atol=0, equal_nan=True)
    assert got["candidate_pairs"] == expected["candidate_pairs"]
