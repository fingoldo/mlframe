"""Regression: ``FeatureTypesConfig.datetime_methods`` lets the user pick
which datetime decomposition columns are emitted. Pre-fix the set was
hardcoded to {day, weekday, month, hour} and no override surface existed.

Per the richness-first policy the default keeps backward-compat (same
four columns) BUT new methods (year / dayofyear / minute / second /
is_weekend) become available via the config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_feature_types_config_accepts_datetime_methods():
    """The field must exist on FeatureTypesConfig (extra='forbid' would
    otherwise reject construction)."""
    from mlframe.training.configs import FeatureTypesConfig

    cfg = FeatureTypesConfig(datetime_methods={"year", "ordinal_day"})
    assert cfg.datetime_methods == {"year", "ordinal_day"}


def test_feature_types_config_default_keeps_backcompat():
    """Default value preserves the historic {day, weekday, month, hour} set."""
    from mlframe.training.configs import FeatureTypesConfig

    cfg = FeatureTypesConfig()
    # New methods should be available as a strict superset; default at
    # least includes the legacy four.
    default = cfg.datetime_methods
    assert {"day", "weekday", "month", "hour"}.issubset(default), (
        f"backward-compat broken: default datetime_methods = {default}"
    )


def test_create_date_features_emits_year_and_dayofyear_when_requested():
    """End-to-end: pass a config that asks for year+dayofyear, verify the
    resulting frame has those columns."""
    from mlframe.feature_engineering.basic import create_date_features

    df = pl.DataFrame({
        "ts": [pd.Timestamp("2024-03-15"), pd.Timestamp("2025-07-04")],
    })
    methods = {"year": np.int32, "ordinal_day": np.int16}
    out = create_date_features(df, cols=["ts"], delete_original_cols=True, methods=methods)
    cols = set(out.columns)
    assert "ts_year" in cols
    assert "ts_ordinal_day" in cols
