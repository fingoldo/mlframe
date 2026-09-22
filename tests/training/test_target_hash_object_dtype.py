"""An object-dtype target cannot be content-hashed, so the hash reports "unknown" instead of hashing pointer bytes."""

import numpy as np

from mlframe.training.pipeline._pipeline_cache import _full_target_content_hash


def test_object_target_reports_unknown_not_a_pointer_hash():
    a = np.array(["x", "y", "x"], dtype=object)
    b = np.array(["x", "y", "x"], dtype=object)  # same content, different element addresses
    assert _full_target_content_hash(a) == ""
    assert _full_target_content_hash(b) == ""


def test_numeric_target_hash_is_content_based():
    a = np.array([1.0, 2.0, 3.0])
    assert _full_target_content_hash(a) == _full_target_content_hash(a.copy()) != ""
    assert _full_target_content_hash(a) != _full_target_content_hash(np.array([1.0, 2.0, 4.0]))
