"""Regression sensors for W16A (S33): extended Kaggle-style defaults + sin/cos cyclical features + multi-tz WARN.

Covers ``create_date_features`` extended defaults and the new ``add_cyclical_date_features``
helper. See ``docs/date_features_kaggle_research.md`` for the rationale behind the chosen
default set and the sin/cos pair contract.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_engineering.basic import (
    _DEFAULT_DATE_METHODS,
    add_cyclical_date_features,
    create_date_features,
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _make_year_pd(n: int = 24) -> pd.DataFrame:
    """Two-year span (~monthly granularity) so each month + quarter shows up at least once."""
    base = pd.Timestamp("2024-01-01")
    return pd.DataFrame({"d": [base + pd.Timedelta(days=15 * i) for i in range(n)]})


def _make_year_pl(n: int = 24) -> pl.DataFrame:
    """Helper: Make year pl."""
    base = datetime(2024, 1, 1)
    return pl.DataFrame({"d": [base + timedelta(days=15 * i) for i in range(n)]})


# --------------------------------------------------------------------------------------
# Extended defaults
# --------------------------------------------------------------------------------------


def test_default_methods_emits_year_week_quarter_isweekend_dayofyear():
    """Extended Kaggle-style defaults must include the new fields on top of the legacy trio."""
    out = create_date_features(_make_year_pd(), cols=["d"], delete_original_cols=False)
    for suffix in ("year", "quarter", "month", "week_of_year", "day", "day_of_year", "weekday", "is_weekend"):
        assert f"d_{suffix}" in out.columns, f"default extraction missing d_{suffix}; got {list(out.columns)}"


def test_year_is_int32_not_int8_overflow_safe():
    """``year`` MUST be int32; int8/int16 would overflow on >=2128 year input."""
    df = pd.DataFrame({"d": pd.to_datetime(["2200-06-01", "2024-01-01"])})
    out = create_date_features(df, cols=["d"], delete_original_cols=False)
    assert out["d_year"].dtype == np.int32
    assert int(out["d_year"].iloc[0]) == 2200


def test_is_weekend_is_bool_and_correct_on_sat_sun():
    """Saturday + Sunday must be True, Mon..Fri False (Mon=0..Sun=6 convention)."""
    dates = pd.to_datetime(
        [
            "2024-01-01",  # Monday
            "2024-01-05",  # Friday
            "2024-01-06",  # Saturday
            "2024-01-07",  # Sunday
        ]
    )
    out = create_date_features(pd.DataFrame({"d": dates}), cols=["d"], delete_original_cols=False)
    assert out["d_is_weekend"].dtype == bool
    assert list(out["d_is_weekend"].astype(bool)) == [False, False, True, True]


def test_quarter_values_in_1_to_4():
    """Quarter must be in {1, 2, 3, 4} across a 2-year span."""
    out = create_date_features(_make_year_pd(48), cols=["d"], delete_original_cols=False)
    qs = set(int(v) for v in out["d_quarter"].values)
    assert qs <= {1, 2, 3, 4}
    assert qs == {1, 2, 3, 4}


def test_week_of_year_in_1_to_53():
    """Week of year in 1 to 53."""
    out = create_date_features(_make_year_pd(48), cols=["d"], delete_original_cols=False)
    weeks = set(int(v) for v in out["d_week_of_year"].values)
    assert all(1 <= w <= 53 for w in weeks)


def test_day_of_year_in_1_to_366():
    """Day of year in 1 to 366."""
    out = create_date_features(_make_year_pd(48), cols=["d"], delete_original_cols=False)
    doy = set(int(v) for v in out["d_day_of_year"].values)
    assert all(1 <= d <= 366 for d in doy)


# --------------------------------------------------------------------------------------
# Cyclical sin/cos
# --------------------------------------------------------------------------------------


def test_cyclical_features_pairs_sin_cos_normalised():
    """sin^2 + cos^2 == 1 invariant for every emitted pair (float32 precision)."""
    df = _make_year_pd(48)
    out = add_cyclical_date_features(df, cols=["d"])
    sin_cols = [c for c in out.columns if c.endswith("_sin")]
    cos_cols = [c for c in out.columns if c.endswith("_cos")]
    assert len(sin_cols) == len(cos_cols) > 0
    # Pair by stripped prefix (sorting independent _sin / _cos lists desynchronises them when prefixes contain ``_o*`` < ``_s*`` < ``_*_sin`` like ``d_day``, ``d_day_of_week``, ``d_day_of_year``).
    cos_by_prefix = {c[:-4]: c for c in cos_cols}
    for sc in sorted(sin_cols):
        prefix = sc[:-4]
        assert prefix in cos_by_prefix, f"sin column {sc!r} has no matching cos pair"
        cc = cos_by_prefix[prefix]
        s = out[sc].to_numpy().astype(np.float64)
        c = out[cc].to_numpy().astype(np.float64)
        # float32 precision -> allow ~1e-6 slack
        np.testing.assert_allclose(s * s + c * c, np.ones_like(s), atol=1e-6)


def test_cyclical_features_dtype_float32():
    """Cyclical features dtype float32."""
    out = add_cyclical_date_features(_make_year_pd(12), cols=["d"])
    for c in out.columns:
        if c.endswith("_sin") or c.endswith("_cos"):
            assert out[c].dtype == np.float32, f"{c} dtype is {out[c].dtype}, expected float32"


def test_cyclical_features_range_minus_one_to_one():
    """Cyclical features range minus one to one."""
    out = add_cyclical_date_features(_make_year_pd(48), cols=["d"])
    for c in out.columns:
        if c.endswith("_sin") or c.endswith("_cos"):
            arr = out[c].to_numpy()
            assert arr.min() >= -1.0 - 1e-6
            assert arr.max() <= 1.0 + 1e-6


def test_cyclical_month_jan_dec_adjacent():
    """Distance(month=12, month=1) ~= distance(month=6, month=7) in the (sin, cos) plane.

    Integer encoding has dist(Jan, Dec)=11 vs dist(Jun, Jul)=1; sin/cos preserves wrap-around.
    """
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(
                [
                    "2024-01-15",  # month=1
                    "2024-06-15",  # month=6
                    "2024-07-15",  # month=7
                    "2024-12-15",  # month=12
                ]
            )
        }
    )
    out = add_cyclical_date_features(
        df,
        cols=["d"],
        periods=(("month", 12.0),),
    )
    sin = out["d_month_sin"].to_numpy().astype(np.float64)
    cos = out["d_month_cos"].to_numpy().astype(np.float64)
    # rows: jan=0, jun=1, jul=2, dec=3
    d_dec_jan = np.sqrt((sin[3] - sin[0]) ** 2 + (cos[3] - cos[0]) ** 2)
    d_jun_jul = np.sqrt((sin[2] - sin[1]) ** 2 + (cos[2] - cos[1]) ** 2)
    # On the unit circle, distance between adjacent months is 2*sin(pi/12) ~= 0.5176 for both pairs.
    np.testing.assert_allclose(d_dec_jan, d_jun_jul, rtol=1e-4)


def test_cyclical_unknown_period_name_raises():
    """Cyclical unknown period name raises."""
    with pytest.raises(ValueError, match="Unknown cyclical period"):
        add_cyclical_date_features(
            _make_year_pd(5),
            cols=["d"],
            periods=(("not_a_period", 1.0),),
        )


def test_cyclical_is_weekend_polars_raises():
    """``is_weekend`` is binary, not periodic; cyclical encoding meaningless -- must raise on polars."""
    df = _make_year_pl(5)
    with pytest.raises(ValueError, match="is_weekend"):
        add_cyclical_date_features(df, cols=["d"], periods=(("is_weekend", 2.0),))


def test_create_date_features_add_cyclical_kwarg_emits_pairs():
    """``add_cyclical=True`` in ``create_date_features`` must add sin/cos columns alongside the scalars."""
    out = create_date_features(
        _make_year_pd(12),
        cols=["d"],
        delete_original_cols=False,
        add_cyclical=True,
        cyclical_periods=(("month", 12.0),),
    )
    assert "d_month" in out.columns
    assert "d_month_sin" in out.columns
    assert "d_month_cos" in out.columns


# --------------------------------------------------------------------------------------
# Multi-tz WARN
# --------------------------------------------------------------------------------------


def test_mixed_tz_warns_with_concrete_tzlist(caplog):
    """Mixed-tz columns must trigger a WARN that lists EVERY observed tz string."""
    utc = pd.to_datetime(["2024-01-01", "2024-06-01"]).tz_localize("UTC")
    ny = pd.to_datetime(["2024-01-01", "2024-06-01"]).tz_localize("America/New_York")
    naive = pd.to_datetime(["2024-01-01", "2024-06-01"])
    df = pd.DataFrame({"a": utc, "b": ny, "c": naive})
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.basic"):
        create_date_features(df, cols=["a", "b", "c"], delete_original_cols=False)
    warns = [r.message for r in caplog.records if r.levelname == "WARNING" and "timezones" in r.message.lower()]
    assert warns, f"Expected tz-mix WARN; got {[r.message for r in caplog.records]}"
    combined = " ".join(warns)
    assert "UTC" in combined
    assert "America/New_York" in combined
    assert "naive" in combined


def test_single_tz_does_not_warn(caplog):
    """No WARN when all cols share one tz (false-positive sensor)."""
    utc1 = pd.to_datetime(["2024-01-01", "2024-06-01"]).tz_localize("UTC")
    utc2 = pd.to_datetime(["2023-12-31", "2025-01-01"]).tz_localize("UTC")
    df = pd.DataFrame({"a": utc1, "b": utc2})
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.basic"):
        create_date_features(df, cols=["a", "b"], delete_original_cols=False)
    warns = [r for r in caplog.records if r.levelname == "WARNING" and "timezones" in r.message.lower()]
    assert not warns, [r.message for r in warns]


# --------------------------------------------------------------------------------------
# Polars / pandas equivalence
# --------------------------------------------------------------------------------------


def test_create_date_features_polars_branch_matches_pandas():
    """Same input -> same numeric output across backends for the extended default set."""
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=11 * i) for i in range(20)]
    pd_df = pd.DataFrame({"d": pd.to_datetime(dates)})
    pl_df = pl.DataFrame({"d": dates})

    pd_out = create_date_features(pd_df, cols=["d"], delete_original_cols=True)
    pl_out = create_date_features(pl_df, cols=["d"], delete_original_cols=True)

    for method in _DEFAULT_DATE_METHODS.keys():
        col = f"d_{method}"
        pd_vals = pd_out[col].to_numpy()
        pl_vals = pl_out[col].to_numpy()
        # bool / int comparison: cast both to int for the diff so numpy 0/1 vs True/False match.
        np.testing.assert_array_equal(
            pd_vals.astype(np.int64),
            pl_vals.astype(np.int64),
            err_msg=f"backend mismatch for {col}: pandas={pd_vals[:5]} polars={pl_vals[:5]}",
        )


def test_add_cyclical_polars_branch_matches_pandas():
    """sin/cos values must agree across backends within float32 precision."""
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=11 * i) for i in range(20)]
    pd_df = pd.DataFrame({"d": pd.to_datetime(dates)})
    pl_df = pl.DataFrame({"d": dates})
    pd_out = add_cyclical_date_features(pd_df, cols=["d"])
    pl_out = add_cyclical_date_features(pl_df, cols=["d"])
    for c in pd_out.columns:
        if c.endswith("_sin") or c.endswith("_cos"):
            np.testing.assert_allclose(
                pd_out[c].to_numpy().astype(np.float64),
                pl_out[c].to_numpy().astype(np.float64),
                atol=1e-5,
                err_msg=f"backend mismatch for {c}",
            )


# --------------------------------------------------------------------------------------
# Research doc
# --------------------------------------------------------------------------------------


def test_kaggle_research_doc_exists():
    """The 1-page research summary must accompany the implementation."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doc_path = os.path.join(repo_root, "docs", "date_features_kaggle_research.md")
    assert os.path.isfile(doc_path), f"missing research doc at {doc_path}"
    with open(doc_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert len(content) > 1500, "research doc looks suspiciously short"
    # Must mention each canonical reference family (loose substring checks; not asserting exact URLs).
    for needle in ("fastai", "scikit-learn", "feature-engine", "cyclical", "sin", "cos", "timezone", "year", "quarter", "week_of_year"):
        assert needle.lower() in content.lower(), f"research doc missing reference to '{needle}'"
