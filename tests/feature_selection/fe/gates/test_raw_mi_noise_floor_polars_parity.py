"""The raw-MI noise floor is the same for a pandas and a polars frame holding the same columns.

Before the fix a polars ``raw_X`` returned a floor of 0.0, so the local MI gate kept every candidate on a polars frame.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_selection.filters._unified_fe_gate import raw_mi_noise_floor


@pytest.fixture
def frame_and_target():
    """Twelve standard-normal columns and a target driven by the first one."""
    rng = np.random.default_rng(0)
    n = 5000
    data = {f"c{i}": rng.standard_normal(n) for i in range(12)}
    y = (data["c0"] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    return data, y


def test_polars_floor_equals_pandas_floor(frame_and_target):
    """The floor of a polars frame equals the floor of the pandas frame built from the same arrays."""
    data, y = frame_and_target
    floor_pd = raw_mi_noise_floor(pd.DataFrame(data), y)
    floor_pl = raw_mi_noise_floor(pl.DataFrame(data), y)
    assert floor_pl > 0.0
    assert floor_pl == pytest.approx(floor_pd, abs=1e-12)


def test_polars_floor_skips_non_numeric_columns(frame_and_target):
    """A string column in a polars frame is left out of the floor, as a non-numeric pandas column is."""
    data, y = frame_and_target
    with_text = {**data, "label": [f"s{i % 3}" for i in range(len(y))]}
    floor_pd = raw_mi_noise_floor(pd.DataFrame(with_text), y)
    floor_pl = raw_mi_noise_floor(pl.DataFrame(with_text), y)
    assert floor_pl == pytest.approx(floor_pd, abs=1e-12)
