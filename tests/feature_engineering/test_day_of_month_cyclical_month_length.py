"""Day-of-month cyclical encoding against the actual month length, as an opt-in beside the fixed-31 default."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_engineering.basic import MONTH_LENGTH_PERIOD, add_cyclical_date_features

_DATES = pd.to_datetime(["2023-02-28", "2023-03-01", "2023-01-31", "2023-02-01"])


def _angle(df, col="d"):
    return np.arctan2(np.asarray(df[f"{col}_day_sin"], dtype=np.float64), np.asarray(df[f"{col}_day_cos"], dtype=np.float64))


def _circular_gap(a, b):
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


@pytest.mark.parametrize("frame", ["pandas", "polars"])
def test_month_end_and_next_month_start_are_adjacent(frame):
    """With a fixed 31, 28 Feb and 1 Mar landed about two thirds of the circle apart."""
    df = pd.DataFrame({"d": _DATES})
    if frame == "polars":
        df = pl.from_pandas(df)
    out = add_cyclical_date_features(df, cols=["d"], periods=(("day", MONTH_LENGTH_PERIOD),))
    angles = _angle(out)
    step_feb = 2 * np.pi / 28
    assert _circular_gap(angles[0], angles[1]) == pytest.approx(step_feb, rel=1e-5), "28 Feb -> 1 Mar is one day"
    assert _circular_gap(angles[2], angles[3]) == pytest.approx(2 * np.pi / 31, rel=1e-5), "31 Jan -> 1 Feb is one day"


def test_the_default_encoding_is_unchanged():
    """Saved models were trained on the fixed-31 encoding and it is recomputed at predict, so the default must not move."""
    out = add_cyclical_date_features(pd.DataFrame({"d": _DATES}), cols=["d"], periods=(("day", 31.0),))
    np.testing.assert_allclose(np.asarray(out["d_day_sin"], dtype=np.float64), np.sin(2 * np.pi * np.array([28, 1, 31, 1]) / 31.0), atol=1e-6)


def test_the_marker_is_rejected_for_other_parts():
    with pytest.raises(ValueError, match="day-of-month only"):
        add_cyclical_date_features(pd.DataFrame({"d": _DATES}), cols=["d"], periods=(("month", MONTH_LENGTH_PERIOD),))
