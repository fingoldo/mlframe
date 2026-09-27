"""A missing classification label is missing in every target derived from it, not a class.

``null_count()`` does not count NaN, and ``NaN >= threshold`` is True in polars: a NaN row used to be labelled 1 under a
lower threshold, and a fillna(0) before that labelled missing rows as a real 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _frame(backend: str, missing):
    data = {"x": [1.0, 2.0, 3.0, 4.0], "hired": [0.0, 2.0, missing, 1.0]}
    return pl.DataFrame(data, strict=False) if backend == "polars" else pd.DataFrame(data)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("missing", [float("nan"), None])
@pytest.mark.parametrize(
    "rule, name, expected",
    [
        ({"classification_lower_thresholds": {"hired": 1}}, "hired_above_1", [0, 1, 1]),
        ({"classification_upper_thresholds": {"hired": 1}}, "hired_below_1", [1, 0, 1]),
        ({"classification_exact_values": {"hired": 2}}, "hired_eq_2", [0, 1, 0]),
        ({}, "hired", [0, 2, 1]),
    ],
)
def test_a_missing_classification_label_stays_missing(backend, missing, rule, name, expected):
    extractor = SimpleFeaturesAndTargetsExtractor(classification_targets=["hired"], **rule)
    y = np.asarray(next(iter(extractor.build_targets(_frame(backend, missing)).values()))[name], dtype=np.float64)
    assert np.isnan(y[2]), f"the unlabelled row became {y[2]!r}"
    np.testing.assert_array_equal(y[[0, 1, 3]], expected)


@pytest.mark.parametrize("backend", ["polars", "pandas"])
def test_a_fully_labelled_classification_target_is_still_int8(backend):
    extractor = SimpleFeaturesAndTargetsExtractor(classification_targets=["hired"], classification_lower_thresholds={"hired": 1})
    y = next(iter(extractor.build_targets(_frame(backend, 0.0)).values()))["hired_above_1"]
    assert np.asarray(y).dtype == np.int8
    np.testing.assert_array_equal(np.asarray(y), [0, 1, 0, 1])


def test_fractional_labelled_values_are_still_refused():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cls": [0.0, 1.5, None]})
    with pytest.raises(ValueError, match="non-integer"):
        SimpleFeaturesAndTargetsExtractor(classification_targets=["cls"]).build_targets(df)
