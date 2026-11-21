"""Cluster 35 regressions: string digests keep element boundaries, and a constant base never sets the time-aware flag (audit 2026-11-20, findings 35-F3, 35-F8)."""

import numpy as np

from mlframe.training.composite.ensemble import _full_array_digest
from mlframe.training.composite.discovery._tiny_rerank import _base_varies


def test_string_digest_separates_element_boundaries():
    """["ab", "c"] and ["a", "bc"] used to join to the same bytes and collide (35-F8)."""
    a = np.array(["ab", "c"])
    b = np.array(["a", "bc"])
    assert _full_array_digest(a) != _full_array_digest(b)


def test_string_digest_is_stable_for_equal_content():
    """Equal arrays still share a digest."""
    assert _full_array_digest(np.array(["x", "yz"])) == _full_array_digest(np.array(["x", "yz"]))


def test_constant_and_zero_bases_do_not_count_as_varying():
    """A constant or all-zero placeholder base is not a time index (35-F3)."""
    assert not _base_varies(np.zeros(10))
    assert not _base_varies(np.full(10, 3.0))
    assert _base_varies(np.arange(10.0))
