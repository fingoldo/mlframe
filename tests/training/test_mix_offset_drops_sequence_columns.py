"""The mix-offset gate's input frame drops embedding (list / array cell) columns."""

import numpy as np
import pandas as pd

from mlframe.training.core._main_train_suite_phases import _without_sequence_columns


def test_list_and_array_columns_are_dropped_and_scalar_ones_kept():
    """Object columns of list / ndarray cells go; numeric, category and string columns stay; a frame without any is returned as it is."""
    n = 5
    df = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "emb_list": [[0.0, 1.0] for _ in range(n)],
            "emb_arr": [np.zeros(3) for _ in range(n)],
            "cat": pd.Categorical(list("abcab")),
            "s": list("vwxyz"),
        }
    )
    out = _without_sequence_columns(df)
    assert list(out.columns) == ["x", "cat", "s"]
    clean = df[["x", "cat", "s"]]
    assert _without_sequence_columns(clean) is clean
