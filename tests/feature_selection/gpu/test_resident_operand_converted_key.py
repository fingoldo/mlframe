"""A resident operand asked for in another dtype is keyed on the caller's array: repeats hit without copying or re-hashing, different content never aliases."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_resident_operands as ro


def test_converted_operand_is_served_from_the_cache(monkeypatch):
    """The same float32 column asked as float64 three times: one hash, one upload, the same device array, the float64 values."""
    ro.clear_fe_resident_operands()
    col = np.random.default_rng(0).standard_normal(5000).astype(np.float32)
    hashed = []
    real = ro._content_hash_memoized

    def counting(a):
        """Record whether the hashed array is the caller's column, then hash it."""
        hashed.append(a is col)
        return real(a)

    monkeypatch.setattr(ro, "_content_hash_memoized", counting)
    first = ro.resident_operand(col, ("gate_op", "c"), dtype=cp.float64)
    again = [ro.resident_operand(col, ("gate_op", "c"), dtype=cp.float64) for _ in range(2)]
    assert len(again) > 0 and all(g is first for g in again)
    assert first.dtype == cp.float64
    np.testing.assert_array_equal(cp.asnumpy(first), col.astype(np.float64))
    assert all(hashed)  # the caller's array is what gets hashed, so the memo can serve it


def test_different_content_or_target_dtype_gets_its_own_entry():
    """Other values, or the same values in another target dtype, are separate uploads with their own bytes."""
    ro.clear_fe_resident_operands()
    a = np.arange(100, dtype=np.float32)
    b = a + 1
    ga = ro.resident_operand(a, "x", dtype=cp.float64)
    gb = ro.resident_operand(b, "x", dtype=cp.float64)
    g32 = ro.resident_operand(a, "x", dtype=cp.float32)
    assert ga is not gb and ga is not g32
    np.testing.assert_array_equal(cp.asnumpy(gb), b.astype(np.float64))
    assert g32.dtype == cp.float32
