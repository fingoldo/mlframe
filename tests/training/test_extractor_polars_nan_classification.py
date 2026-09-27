"""A float NaN in a polars classification column is a missing label, not the positive class.

``null_count()`` does not count NaN, and ``NaN >= threshold`` is True in polars: the row passed the null guard and was
labelled 1 under a lower threshold. It is now caught by the same guard as a null.
"""

from __future__ import annotations

import polars as pl
import pytest

from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


@pytest.mark.parametrize("missing", [float("nan"), None])
def test_a_missing_classification_label_is_not_silently_labelled(missing):
    """A null or NaN classification label raises instead of being thresholded into a class."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "hired": [0.0, 2.0, missing, 1.0]}, strict=False)
    extractor = SimpleFeaturesAndTargetsExtractor(classification_targets=["hired"], classification_lower_thresholds={"hired": 1})
    with pytest.raises(ValueError, match="contains nulls"):
        extractor.build_targets(df)
