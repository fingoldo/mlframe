"""Cluster 28 regression: the narrowing helper's docstring states that NaN features also yield None (audit 2026-11-20, finding 28-F51).

Behaviour pin: a frame with a NaN returns None, and a finite numeric frame returns an array; the docstring says both cases.
"""

import numpy as np
import pandas as pd

from mlframe.training import _uncertainty_eval as uncert


def test_non_finite_values_give_none():
    """A NaN in a numeric column makes the helper return None (28-F51)."""
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    assert uncert._narrow_numeric_frame(df, ["a", "b"]) is None


def test_finite_numeric_frame_gives_an_array():
    """Control: a finite numeric frame returns a float array (28-F51)."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    out = uncert._narrow_numeric_frame(df, ["a", "b"])
    assert out is not None and out.shape == (2, 2)


def test_docstring_mentions_non_finite():
    """The docstring names the non-finite case (28-F51)."""
    assert "non-finite" in (uncert._narrow_numeric_frame.__doc__ or "")
