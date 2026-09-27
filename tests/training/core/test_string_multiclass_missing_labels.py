"""A string multiclass target with missing labels is encoded over its labelled values; the gaps stay missing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.configs import TargetTypes
from mlframe.training.core._main_train_suite_encoding import _encode_string_multiclass_target


def test_missing_labels_are_not_a_class_and_stay_missing():
    """np.unique could not order None against strings, so the suite died before training any target."""
    metadata: dict = {}
    values = pd.Series(["b", None, "a", "c", np.nan, "a"], dtype=object)
    codes = _encode_string_multiclass_target(TargetTypes.MULTICLASS_CLASSIFICATION, "grade", values, metadata)
    assert metadata["target_label_classes"]["grade"] == ["a", "b", "c"]
    assert np.isnan(codes[[1, 4]]).all()
    assert codes[[0, 2, 3, 5]].tolist() == [1.0, 0.0, 2.0, 0.0]


def test_a_fully_labelled_target_still_gets_integer_codes():
    codes = _encode_string_multiclass_target(TargetTypes.MULTICLASS_CLASSIFICATION, "g", np.array(["b", "a"], dtype=object), {})
    assert codes.dtype == np.int64 and codes.tolist() == [1, 0]
