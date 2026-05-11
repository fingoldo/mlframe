"""Tests for ``mlframe.training._format`` helpers.

Locks the contract: adaptive ``format_metric`` keeps ~``ndigits``
significant figures regardless of magnitude, and shim-name strippers
collapse the internal ``WithDMatrixReuse`` / ``WithDatasetReuse`` tags
out of user-facing model names.
"""
from __future__ import annotations

import math

import pytest

from mlframe.training._format import (
    format_metric, strip_shim_suffix, short_model_tag,
)


class TestFormatMetric:
    @pytest.mark.parametrize("value, expected", [
        # Default ndigits=2: large values keep 2 decimal places.
        (11497.4655, "11497.47"),
        (13.93, "13.93"),
        (1.0, "1.00"),
        (-13.93, "-13.93"),
        # Exact-1 boundary: still 2 d.p.
        (1.5, "1.50"),
        # 0 special case: 2 d.p. fixed.
        (0.0, "0.00"),
        (-0.0, "-0.00"),
        # Sub-1 values: widen to maintain sig figs.
        (0.4655, "0.47"),      # 0 leading zeros, ndigits=2 -> 2 d.p.
        (0.05, "0.050"),        # 1 leading zero, ndigits=2 -> 3 d.p.
        (0.0034, "0.0034"),     # 2 leading zeros -> 4 d.p.
        (0.00034, "0.00034"),   # 3 leading zeros -> 5 d.p.
        # Very small values switch to scientific notation (F4 fix 2026-05-11): above 4 leading zeros, decimal widening produces unreadable ".000000029"; sci notation is cleaner.
        (1.23e-5, "0.000012"),  # 4 leading zeros -> 6 d.p., still decimal (boundary)
        (2.9e-8, "2.90e-08"),   # 7 leading zeros -> switch to sci
        (1e-12, "1.00e-12"),    # extreme -> sci
        (-2.9e-8, "-2.90e-08"), # negative tiny -> sci
        # Edge cases.
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (float("nan"), "nan"),
        (None, "None"),
    ])
    def test_default_ndigits_2(self, value, expected):
        assert format_metric(value) == expected

    @pytest.mark.parametrize("value, ndigits, expected", [
        (11497.4655, 3, "11497.466"),
        (11497.4655, 1, "11497.5"),
        (0.4655, 3, "0.466"),
        (0.0034, 1, "0.003"),
        # ndigits=4 (legacy default before the user requested 2):
        # large values get 4 d.p. as expected, small still widen.
        (13.93, 4, "13.9300"),
        (0.0034, 4, "0.003400"),  # ndigits=4 + 2 leading zeros = 6 d.p.
    ])
    def test_custom_ndigits(self, value, ndigits, expected):
        assert format_metric(value, ndigits=ndigits) == expected

    def test_non_numeric_strings_pass_through(self):
        """Non-castable input -> str(input) unchanged."""
        assert format_metric("hello") == "hello"

    def test_integer_input_works(self):
        """int input is coerced to float; default 2 d.p."""
        assert format_metric(42) == "42.00"


class TestStripShimSuffix:
    @pytest.mark.parametrize("name, expected", [
        ("XGBRegressorWithDMatrixReuse", "XGBRegressor"),
        ("XGBClassifierWithDMatrixReuse", "XGBClassifier"),
        ("LGBMRegressorWithDatasetReuse", "LGBMRegressor"),
        ("LGBMClassifierWithDatasetReuse", "LGBMClassifier"),
        ("CatBoostRegressor", "CatBoostRegressor"),  # no shim, unchanged
        ("LinearRegression", "LinearRegression"),    # no shim, unchanged
        ("", ""),
    ])
    def test_strip(self, name, expected):
        assert strip_shim_suffix(name) == expected

    def test_non_string_passthrough(self):
        """Non-string input returned unchanged."""
        assert strip_shim_suffix(None) is None
        assert strip_shim_suffix(42) == 42


class TestShortModelTag:
    @pytest.mark.parametrize("name, expected", [
        ("CatBoostRegressor", "cb"),
        ("CatBoostClassifier", "cb"),
        ("XGBRegressor", "xgb"),
        ("XGBRegressorWithDMatrixReuse", "xgb"),
        ("LGBMRegressor", "lgb"),
        ("LGBMRegressorWithDatasetReuse", "lgb"),
        ("HistGradientBoostingRegressor", "hgb"),
        # Non-tree models: keep full (shim-stripped) name.
        ("LinearRegression", "LinearRegression"),
        ("Ridge", "Ridge"),
    ])
    def test_tag(self, name, expected):
        assert short_model_tag(name) == expected

    def test_instance_input(self):
        """Object input: use type(obj).__name__."""
        class FakeXGB:
            pass
        FakeXGB.__name__ = "XGBRegressorWithDMatrixReuse"
        assert short_model_tag(FakeXGB()) == "xgb"
