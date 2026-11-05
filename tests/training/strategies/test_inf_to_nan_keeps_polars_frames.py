"""The infinity-to-NaN step keeps a polars frame a polars frame, with infinities replaced in its float columns.

On a mixed-dtype polars frame the step passed the frame through untouched, so +/-inf reached the imputer, while the linear and MLP pipelines rely
on this step to remove them. A pandas frame was handled correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.strategies.base import _InfToNaNTransformer


def test_polars_float_columns_lose_their_infinities_and_the_frame_stays_polars():
    """A mixed-dtype polars frame comes back as a polars frame with infinities as null; the string column is untouched."""
    X = pl.DataFrame({"f": [1.0, np.inf, -np.inf, 4.0], "s": ["a", "b", "c", "d"]})
    out = _InfToNaNTransformer().fit_transform(X)
    assert isinstance(out, pl.DataFrame)
    assert out["f"].null_count() == 2
    assert out["f"].is_infinite().sum() == 0
    assert out["s"].to_list() == ["a", "b", "c", "d"]


def test_pandas_frames_keep_the_existing_behaviour():
    """The pandas path is unchanged: infinities become NaN in a pandas frame."""
    X = pd.DataFrame({"f": [1.0, np.inf, 3.0]})
    out = _InfToNaNTransformer().fit_transform(X)
    assert out["f"].isna().sum() == 1
