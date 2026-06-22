"""Regression: report class-label resolution must index the encoder by enumerate
POSITION (class_id), not by the raw integer label VALUE.

``LabelEncoder.classes_`` is positionally ordered. Indexing it by a non-0-based integer
label value picks the wrong class, or raises IndexError when the value exceeds the number
of classes. The fix (and its extracted ``_resolve_class_label`` helper) indexes by the
enumerate position, matching the already-fixed sibling sites.
"""

import numpy as np
import pytest

from mlframe.training.reporting._reporting_probabilistic import _resolve_class_label


class _Enc:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)


def test_non_zero_based_integer_labels_resolve_by_position():
    # Two classes whose RAW labels are 7 and 9 (non-0-based). classes_ holds the
    # original string labels in positional order.
    enc = _Enc(["cat", "dog"])
    # class_id=0 -> "cat", class_id=1 -> "dog". The numeric class_name (7, 9) is the
    # report's stand-in and must NOT be used as the index.
    assert _resolve_class_label(0, 7, enc) == "cat"
    assert _resolve_class_label(1, 9, enc) == "dog"


def test_value_indexing_would_have_raised_indexerror():
    # Demonstrates the pre-fix failure mode: indexing classes_ by the label VALUE 9
    # is out of range for a 2-element classes_ array.
    enc = _Enc(["cat", "dog"])
    with pytest.raises(IndexError):
        _ = enc.classes_[9]
    # The helper avoids it entirely by using class_id.
    assert _resolve_class_label(1, 9, enc) == "dog"


def test_non_numeric_class_name_passthrough():
    assert _resolve_class_label(0, "alpha", _Enc(["x"])) == "alpha"
    # No encoder -> stringify the class_name as-is.
    assert _resolve_class_label(0, 5, None) == "5"
