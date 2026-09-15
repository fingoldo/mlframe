"""Fit-cache key components: a failed content signature never keys on id(), and the process-wide key hashes deep params."""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fingerprints import _content_array_signature


class _Unsignable:
    """An array-like whose shape access raises, forcing the outermost fallback."""

    @property
    def shape(self):
        """Raise, as a broken lazy array might."""
        raise RuntimeError("shape unavailable")


def test_content_array_signature_fallback_never_keys_on_id():
    """Two calls on the SAME object must not return an equal key: an id() key collides once CPython recycles the address."""
    obj = _Unsignable()
    k1 = _content_array_signature(obj)
    k2 = _content_array_signature(obj)
    assert k1[0] == "uncached"
    assert k1 != k2, f"fallback key is stable across calls, so it can match a different object at a recycled address: {k1}"
    assert k1[1] != id(obj)
