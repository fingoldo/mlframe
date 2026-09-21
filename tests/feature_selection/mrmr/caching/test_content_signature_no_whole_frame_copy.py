"""Cache fingerprints must not materialise the whole frame as one dense array (mrmr_audit_2026-09-14 PERIPHERY-4).

``_content_array_signature`` is documented as a cheap sample of 1024 positions, but it called ``X.to_numpy()`` on the WHOLE frame first. On
a mixed-dtype pandas frame that upcasts every column into one dense block, often object - a full-frame allocation made just to read 1024
cells, which on 100+ GB frames is the dominant allocation of the fit. ``_full_x_content_hash`` needs every byte but did the same upcast
instead of hashing column by column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fingerprints import _content_array_signature, _full_x_content_hash


def _mixed(seed=0, n=5000):
    """int, float and categorical columns: the case where to_numpy() upcasts everything into one object block."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "i": rng.integers(0, 100, size=n),
            "f": rng.normal(size=n),
            "c": pd.Categorical(rng.integers(0, 5, size=n)),
        }
    )


@pytest.fixture
def no_whole_frame_to_numpy(monkeypatch):
    """Make any frame-wide DataFrame.to_numpy raise, so the fingerprints must read per column."""

    def _boom(self, *args, **kwargs):
        """Forbid materialising the frame as one array."""
        raise AssertionError("whole-frame DataFrame.to_numpy() called by a fingerprint")

    monkeypatch.setattr(pd.DataFrame, "to_numpy", _boom)


@pytest.mark.usefixtures("no_whole_frame_to_numpy")
def test_content_signature_samples_per_column():
    """The cheap signature must be usable without a frame-wide conversion, and must still tell different content apart."""
    a = _content_array_signature(_mixed(0))
    b = _content_array_signature(_mixed(1))
    assert a[0] != "uncached" and b[0] != "uncached", f"signature fell back to an uncached key: {a[0]!r}"
    assert a != b, "two frames with different content produced the same signature"
    assert a == _content_array_signature(_mixed(0)), "the signature is not deterministic for identical content"


@pytest.mark.usefixtures("no_whole_frame_to_numpy")
def test_full_content_hash_streams_per_column():
    """The full-content hash must be computed without a frame-wide conversion and must distinguish content, names and dtypes."""
    base = _full_x_content_hash(_mixed(0))
    assert base, "the full-content hash gave up (empty digest)"
    assert base == _full_x_content_hash(_mixed(0))
    assert base != _full_x_content_hash(_mixed(1)), "different content, same digest"
    assert base != _full_x_content_hash(_mixed(0).rename(columns={"f": "g"})), "renamed columns, same digest"
    changed_dtype = _mixed(0)
    changed_dtype["i"] = changed_dtype["i"].astype(np.float64)
    assert base != _full_x_content_hash(changed_dtype), "different dtype, same digest"
