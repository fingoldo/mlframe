"""Day-of-month cyclical encoding against the actual month length, as an opt-in beside the fixed-31 default."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_engineering.basic import MONTH_LENGTH_PERIOD, add_cyclical_date_features

_DATES = pd.to_datetime(["2023-02-28", "2023-03-01", "2023-01-31", "2023-02-01"])


def _angle(df, col="d"):
    """Angle on the circle of each row's day-of-month sin/cos encoding."""
    return np.arctan2(np.asarray(df[f"{col}_day_sin"], dtype=np.float64), np.asarray(df[f"{col}_day_cos"], dtype=np.float64))


def _circular_gap(a, b):
    """Shortest distance between two angles on the circle."""
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
    """The month-length period only makes sense for the day of the month; any other part raises."""
    with pytest.raises(ValueError, match="day-of-month only"):
        add_cyclical_date_features(pd.DataFrame({"d": _DATES}), cols=["d"], periods=(("month", MONTH_LENGTH_PERIOD),))


def test_default_encodes_day_against_month_length_and_legacy_version_replays_31():
    """Version 2 (default) makes 28 Feb and 1 Mar neighbours; version 1, replayed for models fitted before versioning, keeps 31."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_engineering.basic import LEGACY_CYCLICAL_ENCODING_VERSION, create_date_features

    df = pd.DataFrame({"t": pd.to_datetime(["2023-02-28", "2023-03-01"])})

    def gap(frame):
        """Euclidean distance between the two rows' (sin, cos) day encodings."""
        a = np.array([frame["t_day_sin"].to_numpy(), frame["t_day_cos"].to_numpy()])
        return float(np.linalg.norm(a[:, 0] - a[:, 1]))

    current = create_date_features(df, cols=["t"], methods={"day": np.int8})
    legacy = create_date_features(df, cols=["t"], methods={"day": np.int8}, cyclical_version=LEGACY_CYCLICAL_ENCODING_VERSION)
    assert gap(current) < 0.3  # one step of a 28-day circle
    assert gap(legacy) > 3 * gap(current)  # day 28 of a 31-day circle vs day 1: four steps apart


def test_predict_replays_the_version_the_model_was_fitted_with():
    """Predict re-derives date features with the encoding version stored at fit: none recorded means version 1."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_engineering.basic import create_date_features
    from mlframe.training.core.predict import _replay_suite_datetime_decomposition

    df = pd.DataFrame({"t": pd.to_datetime(["2023-02-28", "2023-03-01"])})
    meta = {"datetime_methods": {"t": {"day": "int8"}}}
    old = _replay_suite_datetime_decomposition(df.copy(), dict(meta))
    new = _replay_suite_datetime_decomposition(df.copy(), dict(meta, datetime_cyclical_version=2))
    ref_old = create_date_features(df, cols=["t"], methods={"day": np.int8}, cyclical_version=1)
    ref_new = create_date_features(df, cols=["t"], methods={"day": np.int8})
    np.testing.assert_array_equal(old["t_day_sin"].to_numpy(), ref_old["t_day_sin"].to_numpy())
    np.testing.assert_array_equal(new["t_day_sin"].to_numpy(), ref_new["t_day_sin"].to_numpy())


def test_unpickled_extractor_without_a_version_replays_version_1():
    """An extractor pickled before versioning has no attribute and must read as version 1, the encoding it was fitted with."""
    from mlframe.training.extractors._extractors_simple import SimpleFeaturesAndTargetsExtractor

    ex = SimpleFeaturesAndTargetsExtractor.__new__(SimpleFeaturesAndTargetsExtractor)
    assert getattr(ex, "cyclical_version", 1) == 1
