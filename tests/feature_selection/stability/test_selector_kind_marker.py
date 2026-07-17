"""Sensor tests for selector-kind dispatch and related polish items.

Covers fixes A-Low-002 (dedicated ``_mlframe_selector_kind_`` marker stamped by
``_build_pre_pipelines``, consumed by ``_selector_kind``) and A-Low-007 (digest_size
bump from 8 to 16 on ``_selector_params_hash`` for collision robustness).
"""

from __future__ import annotations

import pytest


def test_selector_kind_prefers_explicit_marker_over_class_name():
    """_selector_kind should read ``_mlframe_selector_kind_`` first, falling back
    to class-name heuristics only when the marker is absent."""
    from mlframe.training.core._phase_train_one_target import _selector_kind

    class FakeSelector:
        """Groups tests covering FakeSelector."""
        pass

    sel = FakeSelector()
    # No marker, no recognised class-name suffix -> None.
    assert _selector_kind(sel) is None

    sel._mlframe_selector_kind_ = "MRMR"
    assert _selector_kind(sel) == "MRMR"

    sel._mlframe_selector_kind_ = "RFECV"
    assert _selector_kind(sel) == "RFECV"

    sel._mlframe_selector_kind_ = "BorutaShap"
    assert _selector_kind(sel) == "BorutaShap"

    # Garbage marker values fall through to the class-name branch (returns None here).
    sel._mlframe_selector_kind_ = "Unknown"
    assert _selector_kind(sel) is None


def test_selector_params_hash_uses_16_byte_digest():
    """_selector_params_hash should produce a 32-char hex digest (16 bytes) so the
    birthday-bound collision floor is 2^64, not 2^32 (digest_size=8)."""
    from mlframe.training.core._phase_train_one_target import _selector_params_hash

    class Selector:
        """Groups tests covering Selector."""
        def get_params(self, deep=False):
            """Get params."""
            return {"a": 1, "b": "two"}

    digest = _selector_params_hash(Selector())
    assert digest is not None
    assert isinstance(digest, str)
    assert len(digest) == 32, f"expected 32-char hex (16-byte blake2b), got {len(digest)}: {digest!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
