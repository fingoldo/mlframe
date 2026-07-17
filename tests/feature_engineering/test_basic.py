"""Hypothesis-based tests for basic.py module."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings

from mlframe.feature_engineering.basic import create_date_features


@pytest.mark.parametrize("df_lib", ["pandas", "polars"])
@given(n_rows=st.integers(min_value=1, max_value=100))
@settings(deadline=None)
def test_create_date_features_adds_columns(df_lib, n_rows):
    """Test that date features are added correctly for pandas and polars."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_rows)]
    if df_lib == "pandas":
        df = pd.DataFrame({"date": dates, "value": range(n_rows)})
    else:
        df = pl.DataFrame({"date": dates, "value": list(range(n_rows))})
    result = create_date_features(df, ["date"])

    assert "date_day" in result.columns
    assert "date_weekday" in result.columns
    assert "date_month" in result.columns
    assert "date" not in result.columns  # Original deleted


def test_create_date_features_keep_original():
    """Test keeping original columns."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"date": dates})
    result = create_date_features(df, ["date"], delete_original_cols=False)

    assert "date" in result.columns
    assert "date_day" in result.columns


def test_create_date_features_empty_cols():
    """Test with empty columns list."""
    df = pd.DataFrame({"date": [datetime(2023, 1, 1)], "value": [1]})
    result = create_date_features(df, [])

    # Should return unchanged
    assert list(result.columns) == list(df.columns)


def test_create_date_features_custom_methods():
    """Test with custom methods."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"date": dates})

    result = create_date_features(df, ["date"], methods={"day": np.int16, "month": np.int16})

    assert "date_day" in result.columns
    assert "date_month" in result.columns
    assert "date_weekday" not in result.columns


@pytest.mark.parametrize("df_lib", ["pandas", "polars"])
def test_create_date_features_weekday_range(df_lib):
    """Test that weekday values are in correct range (0-6)."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(7)]
    if df_lib == "pandas":
        df = pd.DataFrame({"date": dates})
    else:
        df = pl.DataFrame({"date": dates})
    result = create_date_features(df, ["date"])

    weekdays = result["date_weekday"].to_list() if df_lib == "polars" else result["date_weekday"].values.tolist()
    assert all(0 <= w <= 6 for w in weekdays)


def test_create_date_features_multiple_cols():
    """Test with multiple date columns."""
    dates1 = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    dates2 = [datetime(2023, 6, 1) + timedelta(days=i) for i in range(5)]
    df = pd.DataFrame({"date1": dates1, "date2": dates2})

    result = create_date_features(df, ["date1", "date2"])

    assert "date1_day" in result.columns
    assert "date2_day" in result.columns
    assert "date1" not in result.columns
    assert "date2" not in result.columns


def test_create_date_features_day_values():
    """Test that day values are correct."""
    dates = [datetime(2023, 1, i) for i in range(1, 11)]
    df = pd.DataFrame({"date": dates})
    result = create_date_features(df, ["date"])

    expected_days = list(range(1, 11))
    assert list(result["date_day"].values) == expected_days


def test_create_date_features_month_values():
    """Test that month values are correct."""
    dates = [datetime(2023, i, 1) for i in range(1, 13)]
    df = pd.DataFrame({"date": dates})
    result = create_date_features(df, ["date"])

    expected_months = list(range(1, 13))
    assert list(result["date_month"].values) == expected_months


def test_create_date_features_invalid_df():
    """Test with invalid dataframe type."""
    with pytest.raises(ValueError):
        create_date_features({"not": "a dataframe"}, ["date"])


@pytest.mark.parametrize("df_lib", ["pandas", "polars"])
def test_create_date_features_dtypes(df_lib):
    """Test that dtypes are correctly applied."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    if df_lib == "pandas":
        df = pd.DataFrame({"date": dates})
        expected_dtype = np.int8
    else:
        df = pl.DataFrame({"date": dates})
        expected_dtype = pl.Int8
    result = create_date_features(df, ["date"])

    assert result["date_day"].dtype == expected_dtype
    assert result["date_weekday"].dtype == expected_dtype
    assert result["date_month"].dtype == expected_dtype


# -----------------------------------------------------------------
# 2026-04-19 — silent column-overwrite WARN sensor
# -----------------------------------------------------------------
# Pre-fix: create_date_features did `df[col + "_" + method] = ...`
# with no collision check. A user column `date_year` (say, fiscal
# year engineered upstream) got silently overwritten with calendar
# year. No warning, no log line — data corruption.


def test_create_date_features_warns_on_column_clash_pandas(caplog):
    """A pre-existing user column colliding with a generated pandas date field must emit an OVERWRITTEN warning."""
    import logging
    from datetime import datetime

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    df = pd.DataFrame(
        {
            "date": dates,
            "date_day": [999, 999, 999, 999, 999],  # user's column, will clash
        }
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.basic"):
        create_date_features(df, ["date"])
    warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("date_day" in m and ("OVERWRITTEN" in m or "overwritten" in m.lower()) for m in warns), (
        f"Expected WARN naming 'date_day' as overwritten; got: {warns}"
    )


def test_create_date_features_warns_on_column_clash_polars(caplog):
    """Same column-clash warning contract as the pandas test, exercised on the polars input path."""
    import logging
    from datetime import datetime

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    df = pl.DataFrame(
        {
            "date": dates,
            "date_month": [99, 99, 99, 99, 99],
        }
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.basic"):
        create_date_features(df, ["date"])
    warns = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("date_month" in m for m in warns), warns


def test_create_date_features_no_warn_without_clash(caplog):
    """False-positive sensor: clean input must not warn (this runs on
    every pipeline; false positives would spam logs)."""
    import logging
    from datetime import datetime

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    df = pd.DataFrame({"date": dates, "other_feature": [1.0] * 5})
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.basic"):
        create_date_features(df, ["date"])
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warns, f"Clean input must not warn; got: {[r.message for r in warns]}"


class TestResolvePandasMethod:
    """Guards the getattr-once date-field resolver (extracts each pandas .dt
    field once instead of hasattr-then-getattr extracting it twice)."""

    def _dt(self):
        """Build a 300-point hourly-stride pandas datetime accessor shared by the resolver tests."""
        return pd.Series(pd.date_range("2021-03-01", periods=300, freq="7h")).dt

    @pytest.mark.parametrize(
        "method",
        [
            "hour",
            "day",
            "weekday",
            "month",
            "day_of_year",
            "is_weekend",
            "week_of_year",
            "year",
            "quarter",
        ],
    )
    def test_resolved_field_matches_direct_accessor(self, method):
        """_resolve_pandas_method must reproduce the exact values of the direct .dt accessor for every supported method."""
        from mlframe.feature_engineering.basic import _resolve_pandas_method, _DATE_METHOD_ALIASES

        dt = self._dt()
        got = _resolve_pandas_method(dt, method, np.float64).to_numpy()
        # Reference field via the documented alias / special-case semantics.
        if method == "is_weekend":
            ref = (dt.weekday >= 5).astype(np.float64).to_numpy()
        elif method == "week_of_year":
            ref = dt.isocalendar().week.astype(np.float64).to_numpy()
        else:
            pd_name = _DATE_METHOD_ALIASES.get(method, (method, None))[0] or method
            ref = getattr(dt, pd_name).astype(np.float64).to_numpy()
        assert np.array_equal(got, ref, equal_nan=True)

    def test_unknown_accessor_raises_valueerror(self):
        """An unrecognized method name must raise ValueError instead of silently returning garbage."""
        from mlframe.feature_engineering.basic import _resolve_pandas_method

        with pytest.raises(ValueError, match="Unknown pandas .dt accessor"):
            _resolve_pandas_method(self._dt(), "not_a_real_field", np.float64)


def test_cyclical_pass_reuses_precomputed_date_fields_not_redecode(monkeypatch):
    """create_date_features must NOT re-decode .dt for cyclical periods whose integer field
    it already extracted (day/weekday/month/day_of_year overlap the default cyclical periods).
    Pre-fix the pandas cyclical branch called _resolve_pandas_method once per cyclical period,
    re-walking the int64-ns array; the fix reuses the already-extracted integer column.
    Spy on _resolve_pandas_method: with the default methods+periods only `hour` (the single
    cyclical period absent from the integer methods) may be resolved during the cyclical pass."""
    import mlframe.feature_engineering.basic as basic

    dates = [datetime(2023, 1, 1) + timedelta(hours=i * 7) for i in range(200)]
    df = pd.DataFrame({"ts": pd.to_datetime(dates)})

    calls = []
    real = basic._resolve_pandas_method

    def spy(series_dt, method, dtype):
        """Record each (method, dtype-kind) pair resolved during the cyclical pass, then delegate to the real resolver."""
        calls.append((method, np.dtype(dtype).kind))
        return real(series_dt, method, dtype)

    monkeypatch.setattr(basic, "_resolve_pandas_method", spy)
    out = basic.create_date_features(df, cols=["ts"])

    # Float (kind 'f') resolutions happen ONLY in the cyclical pass. With reuse, the only
    # cyclical period not already extracted as an integer field is `hour`.
    float_methods = sorted({m for m, k in calls if k == "f"})
    assert float_methods == ["hour"], (
        f"cyclical pass re-decoded already-extracted fields {float_methods}; expected only 'hour' to need a fresh float extraction"
    )
    assert "ts_month_sin" in out.columns and "ts_hour_cos" in out.columns


def test_cyclical_reuse_bit_identical_to_fresh_extraction():
    """The reuse path must be byte-identical to recomputing every cyclical base from .dt."""
    dates = [datetime(2021, 3, 4) + timedelta(hours=i * 5) for i in range(5000)]
    df = pd.DataFrame({"ts": pd.to_datetime(dates)})

    with_reuse = create_date_features(df, cols=["ts"])
    # Force the no-reuse path by calling the cyclical helper standalone (no _precomputed_bases).
    from mlframe.feature_engineering.basic import add_cyclical_date_features

    base = create_date_features(df, cols=["ts"], add_cyclical=False, delete_original_cols=False)
    fresh = add_cyclical_date_features(base, cols=["ts"], delete_original_cols=True)

    cyc_cols = [c for c in with_reuse.columns if c.endswith("_sin") or c.endswith("_cos")]
    for c in cyc_cols:
        np.testing.assert_array_equal(
            with_reuse[c].to_numpy(),
            fresh[c].to_numpy(),
            err_msg=f"cyclical reuse diverged from fresh extraction on {c}",
        )


def test_cyclical_sincos_parallel_bit_identical_to_serial():
    """The prange twin must be byte-identical to the serial loop (each element is independent, no reduction) across hostile inputs: negatives, ties, zero,
    large magnitudes. Guards against a future 'just always parallelize' or a fastmath/reduction-order change that would diverge."""
    from mlframe.feature_engineering.basic import _cyclical_sincos_serial, _cyclical_sincos_parallel

    rng = np.random.default_rng(7)
    base = np.concatenate([rng.standard_normal(20000) * 1000.0, -np.arange(20000.0), np.full(20000, 7.0), np.zeros(10)])
    for scale in (2 * np.pi / 365.0, 0.137, 17.3):
        s_ser, c_ser = _cyclical_sincos_serial(base, scale)
        s_par, c_par = _cyclical_sincos_parallel(base, scale)
        np.testing.assert_array_equal(s_ser, s_par, err_msg=f"sin diverged at scale={scale}")
        np.testing.assert_array_equal(c_ser, c_par, err_msg=f"cos diverged at scale={scale}")


def test_cyclical_sincos_njit_dispatches_to_parallel_above_threshold(monkeypatch):
    """The public dispatcher routes large arrays to the prange twin and small arrays to the serial kernel. Catches a regression that drops the parallel path
    (silently losing the ~12x large-n win) or that lowers the threshold so tiny arrays pay the prange thread-launch floor."""
    import mlframe.feature_engineering.basic as basic

    monkeypatch.setattr(basic, "_CYCLICAL_PAR_THRESHOLD", 1000)
    calls = {"serial": 0, "parallel": 0}
    orig_ser, orig_par = basic._cyclical_sincos_serial, basic._cyclical_sincos_parallel

    def _spy_ser(b, s):
        """Count a call into the serial cyclical kernel, then delegate to it."""
        calls["serial"] += 1
        return orig_ser(b, s)

    def _spy_par(b, s):
        """Count a call into the parallel cyclical kernel, then delegate to it."""
        calls["parallel"] += 1
        return orig_par(b, s)

    monkeypatch.setattr(basic, "_cyclical_sincos_serial", _spy_ser)
    monkeypatch.setattr(basic, "_cyclical_sincos_parallel", _spy_par)

    basic._cyclical_sincos_njit(np.arange(100.0), 0.1)
    basic._cyclical_sincos_njit(np.arange(5000.0), 0.1)
    assert calls == {"serial": 1, "parallel": 1}
