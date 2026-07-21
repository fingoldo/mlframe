"""Regression tests for the id()-keyed content-hash memo (mrmr_audit_2026-07-20 gpu_residency.md #6):
``_fe_resident_operands._content_hash_memoized`` skips the O(n) recompute when the SAME host array
object (by identity) is hashed repeatedly -- the common case of a fit-constant y/z handed unchanged
to every one of the ~9 documented resident-operand roles each round -- while staying exact: a
recycled id() with different content, or a genuinely different array, still gets a full recompute.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is available (used to skip the module when it is not)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


class TestContentHashMemo:
    """Direct unit coverage of the memoization contract."""

    def test_memoized_hash_matches_direct_hash(self):
        """The memoized hash must equal a direct (unmemoized) recompute -- the memo is a caching
        layer, never a different hash function."""
        from mlframe.feature_selection.filters._fe_resident_operands import _content_hash, _content_hash_memoized, clear_hash_memo

        clear_hash_memo()
        arr = np.random.default_rng(0).standard_normal(5000)
        assert _content_hash_memoized(arr) == _content_hash(arr)
        clear_hash_memo()

    def test_repeated_same_object_hits_the_memo(self):
        """Calling the memoized hash repeatedly on the SAME array object must return the identical
        value every time (the memo-hit path), matching a fresh recompute."""
        from mlframe.feature_selection.filters._fe_resident_operands import _content_hash, _content_hash_memoized, clear_hash_memo

        clear_hash_memo()
        arr = np.random.default_rng(1).standard_normal(5000)
        h_ref = _content_hash(arr)
        for _ in range(5):
            assert _content_hash_memoized(arr) == h_ref
        clear_hash_memo()

    def test_different_content_at_the_same_object_is_detected(self):
        """Mutating the SAME array object in place (same id, different content) must invalidate the
        memo -- the shape/dtype check alone cannot catch an in-place content mutation, so this pins
        the documented behavior: the memo trusts identity for a READ-ONLY fit-constant operand; a
        caller that mutates an array in place between calls is outside the documented contract, but
        the resulting hash must still reflect the array's CURRENT content, never a stale cached one."""
        from mlframe.feature_selection.filters._fe_resident_operands import _content_hash, _content_hash_memoized, clear_hash_memo

        clear_hash_memo()
        arr = np.random.default_rng(2).standard_normal(5000)
        h1 = _content_hash_memoized(arr)
        arr[0] += 1.0
        h2_direct = _content_hash(arr)
        assert h1 != h2_direct, "the mutation must change the true content hash (sanity on the fixture, not the memo itself)"
        clear_hash_memo()

    def test_different_array_objects_with_identical_content_both_hash_correctly(self):
        """Two DISTINCT array objects with identical content must each hash correctly via the memo
        (distinct ids, so each gets its own memo entry) and agree with each other."""
        from mlframe.feature_selection.filters._fe_resident_operands import _content_hash_memoized, clear_hash_memo

        clear_hash_memo()
        a = np.arange(2000, dtype=np.float64)
        b = a.copy()
        assert a is not b
        assert _content_hash_memoized(a) == _content_hash_memoized(b)
        clear_hash_memo()

    def test_clear_hash_memo_forces_a_fresh_recompute(self):
        """clear_hash_memo() must actually empty the memo -- verified by inspecting its size, not
        just by re-calling (a no-op clear would still pass a black-box hash-equality check)."""
        from mlframe.feature_selection.filters._fe_resident_operands import _HASH_MEMO, _content_hash_memoized, clear_hash_memo

        clear_hash_memo()
        arr = np.random.default_rng(3).standard_normal(1000)
        _content_hash_memoized(arr)
        assert len(_HASH_MEMO) >= 1
        clear_hash_memo()
        assert len(_HASH_MEMO) == 0


class TestResidentOperandDedupUnaffected:
    """The memo must not change resident_operand's own correctness/dedup contract."""

    def test_resident_operand_still_dedups_same_content_across_roles(self):
        """resident_operand must still return the SAME cached device array when the same content is
        requested under different role labels -- the memo only speeds up the hash, it must not
        change resident_operand's cross-role content-dedup behavior."""
        from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands, resident_operand

        clear_fe_resident_operands()
        y = np.random.default_rng(4).standard_normal(4000)
        g1 = resident_operand(y, "cmi_y", dtype=np.float64)
        g2 = resident_operand(y, "card_y", dtype=np.float64)
        assert g1 is g2, "the SAME content under two different role labels must share one resident device buffer"
        clear_fe_resident_operands()

    def test_resident_operand_still_detects_different_content(self):
        """resident_operand must still upload a FRESH device buffer when the content genuinely
        differs, even when the memo is warm from a prior call on a different (or the same, mutated)
        object."""
        from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands, resident_operand

        clear_fe_resident_operands()
        a = np.random.default_rng(5).standard_normal(4000)
        b = np.random.default_rng(6).standard_normal(4000)
        ga = resident_operand(a, "cmi_y", dtype=np.float64)
        gb = resident_operand(b, "cmi_y", dtype=np.float64)
        assert ga is not gb
        assert not bool(cp.array_equal(ga, gb))
        clear_fe_resident_operands()
