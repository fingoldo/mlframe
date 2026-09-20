"""An unhashable target must force the CatBoost Pool label swap, not silently skip it.

`_full_target_content_hash` returns "" when it cannot hash a target, and both Pool caches compared signatures with a
plain `!=`: two DIFFERENT targets that both failed hashing compared equal, so the cached Pool kept the previous
target's labels. CatBoost then fitted target B's model on target A's labels - and on the val Pool, early-stopped
against them - reporting the metrics under target B's name with no error.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.pipeline import _full_target_content_hash, target_label_changed


@pytest.mark.parametrize("last, new", [("", ""), ("abc", ""), ("", "abc"), (None, "abc"), ("abc", None)])
def test_an_unknown_signature_forces_the_swap(last, new):
    assert target_label_changed(last, new) is True


def test_equal_signatures_reuse_the_cached_label():
    assert target_label_changed("abc", "abc") is False


def test_different_signatures_swap():
    assert target_label_changed("abc", "def") is True


def test_two_distinct_unhashable_targets_do_not_look_identical():
    """The live shape of the bug: the helper yields "" for both, and "" == "" used to mean "same target"."""

    class _Unhashable:
        def __len__(self):
            return 3

        def to_numpy(self):
            raise TypeError("this target cannot be converted")

    a, b = _Unhashable(), _Unhashable()
    sig_a, sig_b = _full_target_content_hash(a), _full_target_content_hash(b)
    assert sig_a == sig_b == "", "this test only means anything while the helper fails on these objects"
    assert target_label_changed(sig_a, sig_b) is True


def test_two_distinct_hashable_targets_are_separated():
    a = np.array([0, 1, 1, 0], dtype=np.int64)
    b = np.array([1, 0, 0, 1], dtype=np.int64)
    assert target_label_changed(_full_target_content_hash(a), _full_target_content_hash(b)) is True
    assert target_label_changed(_full_target_content_hash(a), _full_target_content_hash(a.copy())) is False


def test_an_object_dtype_target_is_reported_as_unknown():
    """Hashing an object array reads POINTER bytes, so the digest describes where the elements live rather than what
    they are: the same target hashes differently between runs and a recycled address can make two different targets
    hash the same. Unknown is the only answer the cache can act on safely."""
    a = np.array(["x", "y", None], dtype=object)
    assert _full_target_content_hash(a) == ""
    assert target_label_changed(_full_target_content_hash(a), _full_target_content_hash(a)) is True
