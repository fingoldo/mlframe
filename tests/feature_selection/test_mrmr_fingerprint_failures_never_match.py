"""A fingerprint the code could not fully compute must never match another one.

MRMR skips a fit when the frame's fingerprint matches a cached one. When a column's cell sample could not be read, the
sample fell back to an empty tuple, so two frames of the same shape that differ only in that column shared a
fingerprint; an ndarray parameter whose content hash failed fell back to repr(), which numpy summarises. Both now give
a never-matching token: the cache misses instead of hitting wrongly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fingerprints import _hashable_params_signature, _mrmr_compute_x_fingerprint


class _UnreadableColumn(pd.DataFrame):
    """A frame whose column "b" cannot be read, as a broken extension dtype or a lazy source can fail."""

    @property
    def _constructor(self):
        return _UnreadableColumn

    def __getitem__(self, key):
        if key == "b":
            raise RuntimeError("column b cannot be materialised")
        return super().__getitem__(key)


def test_an_unreadable_column_makes_the_fingerprint_unmatchable():
    frame = _UnreadableColumn({"a": np.arange(50.0), "b": np.arange(50.0)})
    assert _mrmr_compute_x_fingerprint(frame) != _mrmr_compute_x_fingerprint(frame)


def test_a_readable_frame_keeps_a_stable_fingerprint():
    frame = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)})
    assert _mrmr_compute_x_fingerprint(frame) == _mrmr_compute_x_fingerprint(frame.copy())


class _NoBytes(np.ndarray):
    def tobytes(self, *a, **k):
        raise MemoryError("simulated")


def test_an_array_parameter_whose_content_hash_fails_never_matches():
    arr = np.arange(2000.0).view(_NoBytes)
    assert _hashable_params_signature({"w": arr}) != _hashable_params_signature({"w": arr})
