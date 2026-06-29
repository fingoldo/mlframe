"""Regression: the fit-constant FE operand resident-device cache (``_fe_resident_operands``).

Under ``MLFRAME_FE_GPU_STRICT`` the GPU twins re-uploaded the SAME host fit-constant operands (the label
``y``, the conditioning support ``z``, the base feature columns) on every call. ``resident_operand`` uploads
each ONCE per fit and returns the cached device array on subsequent calls (selection-equivalent: same data,
just not re-uploaded), cleared at FE-step teardown alongside the mempool free + the cmi resident cache.

The load-bearing correctness invariant is the CONTENT FINGERPRINT recycled-id guard: ``id()`` is reused by the
allocator after the parent is GC'd, so a stale buffer must never alias a same-role, same-shape operand that
holds DIFFERENT values. These tests pin: (1) same content -> ONE upload (cache hit); (2) different content
under the SAME role + shape -> NO alias (the second call returns the second array's values, not the first's);
(3) clearing empties the table.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_resident_operands as R


@pytest.fixture(autouse=True)
def _clear():
    R.clear_fe_resident_operands()
    yield
    R.clear_fe_resident_operands()


def test_same_content_hits_uploaded_once(monkeypatch):
    """Two calls with the SAME role + identical (freshly re-allocated) content upload exactly ONCE."""
    n_uploads = {"n": 0}
    orig = cp.asarray

    def spy(a, *args, **kw):
        if not isinstance(a, cp.ndarray):
            n_uploads["n"] += 1
        return orig(a, *args, **kw)

    monkeypatch.setattr(cp, "asarray", spy)

    y = np.arange(1000, dtype=np.int64)
    g1 = R.resident_operand(y, "y_test", dtype=np.int64)
    # fresh host object (different id), identical values -> must still hit the cache (keyed on role+content)
    g2 = R.resident_operand(np.arange(1000, dtype=np.int64), "y_test", dtype=np.int64)
    assert g1 is g2, "identical content + role must return the SAME cached device array (one upload)"
    assert n_uploads["n"] == 1, f"fit-constant uploaded {n_uploads['n']}x; expected 1 (resident)"


def test_recycled_id_different_content_does_not_alias():
    """A same-role, same-shape operand with DIFFERENT values must NOT return the stale device buffer.

    This is the recycled-id silent-correctness guard: without the content fingerprint, a second array that
    happened to reuse the first's identity/role/shape would alias the first's stale device copy. With the
    fingerprint it misses and re-uploads, so the returned values match the SECOND array.
    """
    a = np.zeros(512, dtype=np.int64)
    b = np.arange(512, dtype=np.int64)  # same role + shape + dtype, DIFFERENT values
    ga = R.resident_operand(a, "y_test", dtype=np.int64)
    gb = R.resident_operand(b, "y_test", dtype=np.int64)
    assert ga is not gb, "different content must NOT alias the cached buffer"
    np.testing.assert_array_equal(cp.asnumpy(ga), a)
    np.testing.assert_array_equal(cp.asnumpy(gb), b), "must return the SECOND array's values, never the stale first"


def test_dtype_folded_into_key():
    """The operand is cached in the requested final dtype, and shape/dtype separate entries within a role."""
    y = np.arange(64, dtype=np.int32)
    g = R.resident_operand(y, "y_test", dtype=np.int64)
    assert g.dtype == cp.int64, "operand must be cached in the requested final dtype (collapses repeated astype)"


def test_clear_empties_table():
    R.resident_operand(np.arange(8, dtype=np.int64), "y_test", dtype=np.int64)
    assert len(R._FE_RESIDENT_OPERANDS) >= 1
    R.clear_fe_resident_operands()
    assert len(R._FE_RESIDENT_OPERANDS) == 0


def test_teardown_clears_resident_operands(monkeypatch):
    """The FE-step mempool teardown must drop the resident operand cache (so blocks can be reclaimed)."""
    from mlframe.feature_selection.filters._mrmr_fe_step._step_core import _free_gpu_fe_mempool

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    R.resident_operand(np.arange(8, dtype=np.int64), "y_test", dtype=np.int64)
    assert len(R._FE_RESIDENT_OPERANDS) >= 1
    _free_gpu_fe_mempool()
    assert len(R._FE_RESIDENT_OPERANDS) == 0, "FE-step teardown must clear the resident operand cache"


def test_disable_switch_forces_fresh_upload(monkeypatch):
    """MLFRAME_FE_RESIDENT_OPERANDS=0 (diagnostic A/B) bypasses the cache -> a fresh device array each call."""
    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    y = np.arange(32, dtype=np.int64)
    g1 = R.resident_operand(y, "y_test", dtype=np.int64)
    g2 = R.resident_operand(y, "y_test", dtype=np.int64)
    assert g1 is not g2, "disable switch must force a fresh upload (no caching)"
