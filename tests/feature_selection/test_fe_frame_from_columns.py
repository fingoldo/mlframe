"""The no-copy frame over the leg arrays has the values, dtypes and column order of ``pd.DataFrame(dict)``, holds the arrays themselves and handles an empty dict."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_frame_ops import fe_frame_from_columns


def test_same_frame_as_the_dict_constructor_without_copying():
    """Mixed dtypes, dict order kept, the arrays shared with the dict."""
    rng = np.random.default_rng(0)
    cols = {f"c{i}": rng.standard_normal(500).astype(np.float32 if i % 3 else np.float64) for i in range(20)}
    idx = pd.RangeIndex(500)
    ref = pd.DataFrame(cols, index=idx)
    got = fe_frame_from_columns(cols, idx)
    assert list(got.columns) == list(ref.columns) and list(got.dtypes) == list(ref.dtypes)
    pd.testing.assert_frame_equal(got, ref)
    assert np.shares_memory(got["c3"].to_numpy(), cols["c3"])


def test_empty_dict_and_foreign_index():
    """No columns gives an empty frame on the index; a non-range index is kept."""
    idx = pd.Index([10, 20, 30])
    assert fe_frame_from_columns({}, idx).shape == (3, 0)
    got = fe_frame_from_columns({"a": np.arange(3.0)}, idx)
    assert list(got.index) == [10, 20, 30] and got["a"].tolist() == [0.0, 1.0, 2.0]
