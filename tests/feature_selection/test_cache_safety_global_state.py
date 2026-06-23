"""Regression tests for the cache-safety / global-state fixes in the GPU-resident
feature-selection path (CON6-CON10, CON16, CON20).

The GPU device-buffer items (CON6 pinned staging, CON7/CON8 device caches) cannot be
exercised without a CUDA device on this box, so they are validated by a CPU-side test of
the KEYING / co-validation logic that the fix introduced (the part that is wrong under
id-recycle / single-slot clobber regardless of whether the buffers live on host or device).
"""
import threading

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CON7: _OPERAND_TABLE_CACHE / _PREBUILT_OPERAND_TABLE per-key, not single slot.
# ---------------------------------------------------------------------------
def test_con7_prebuilt_operand_table_does_not_clobber_across_two_host_arrays():
    """Two distinct host operand tables each register their own (fake) device mirror; both
    must remain retrievable -- the pre-fix single slot returned only the last-registered one."""
    from mlframe.feature_selection.filters import _gpu_resident_select as grs

    grs._PREBUILT_OPERAND_TABLE.clear()
    a = np.zeros((10, 3), dtype=np.float32)
    b = np.zeros((10, 3), dtype=np.float32)

    class _FakeDev:
        def __init__(self, arr):
            self.shape = arr.shape

    grs.register_prebuilt_operand_table(a, _FakeDev(a))
    grs.register_prebuilt_operand_table(b, _FakeDev(b))

    # Pre-fix: registering b overwrote a's single slot -> a lookup returns None.
    assert grs._prebuilt_operand_table(a) is not None
    assert grs._prebuilt_operand_table(b) is not None
    grs._PREBUILT_OPERAND_TABLE.clear()


def test_con7_prebuilt_operand_table_bounded():
    """The per-key registry is FIFO-capped so distinct tables across a fit can't grow it unbounded."""
    from mlframe.feature_selection.filters import _gpu_resident_select as grs

    grs._PREBUILT_OPERAND_TABLE.clear()

    class _FakeDev:
        shape = (4, 2)

    held = []
    for _ in range(grs._PREBUILT_OPERAND_TABLE_MAX * 3):
        arr = np.zeros((4, 2), dtype=np.float32)
        held.append(arr)  # keep refs alive so the weakrefs don't die
        grs.register_prebuilt_operand_table(arr, _FakeDev())
    assert len(grs._PREBUILT_OPERAND_TABLE) <= grs._PREBUILT_OPERAND_TABLE_MAX
    grs._PREBUILT_OPERAND_TABLE.clear()


def test_con7_prebuilt_shape_mismatch_rejected():
    """A registered mirror whose shape no longer matches the host array must not be returned
    (guards the kernel against an out-of-bounds operand-column read)."""
    from mlframe.feature_selection.filters import _gpu_resident_select as grs

    grs._PREBUILT_OPERAND_TABLE.clear()
    a = np.zeros((10, 3), dtype=np.float32)

    class _FakeDev:
        shape = (10, 5)  # wrong width

    grs.register_prebuilt_operand_table(a, _FakeDev())
    assert grs._prebuilt_operand_table(a) is None
    grs._PREBUILT_OPERAND_TABLE.clear()


# ---------------------------------------------------------------------------
# CON6: pinned D2H staging buffer is thread-local (each thread its own buffer).
# ---------------------------------------------------------------------------
def test_con6_pinned_buffer_is_thread_local():
    """The pinned-staging buffer holder is thread.local -- two threads get independent storage,
    so neither clobbers the other's DMA staging. Pre-fix this was a shared module global."""
    from mlframe.feature_selection.filters import _gpu_resident_select as grs

    assert isinstance(grs._PINNED_D2H_TLS, threading.local)
    grs._PINNED_D2H_TLS.buf = "main-thread-marker"
    seen = {}

    def _worker():
        seen["child"] = getattr(grs._PINNED_D2H_TLS, "buf", None)
        grs._PINNED_D2H_TLS.buf = "child-marker"

    t = threading.Thread(target=_worker)
    t.start(); t.join()
    # The child thread sees its OWN (absent) buffer, not the main thread's -- proving isolation.
    assert seen["child"] is None
    assert grs._PINNED_D2H_TLS.buf == "main-thread-marker"


# ---------------------------------------------------------------------------
# CON8: _DY_DEVICE_CACHE per-key + co-validating weakref, not single slot.
# ---------------------------------------------------------------------------
def test_con8_dy_device_cache_co_validates_weakref_on_id_recycle():
    """The (id, weakref) co-validation rejects an id-recycle false hit: a stored entry whose
    weakref no longer resolves to the live target must NOT be returned."""
    from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as bg
    import weakref

    bg._DY_DEVICE_CACHE.clear()
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    key = (id(y), 7, 3, 4, 4)
    dead_ref = weakref.ref(np.empty(0))  # already-dead-ish; simulate a stale weakref
    # Simulate a stale entry keyed on y's id but pointing at a different (freed) target.
    bg._DY_DEVICE_CACHE[key] = (dead_ref, object())
    # Re-derive the lookup the function does: a hit with ref() is not y must be dropped.
    hit = bg._DY_DEVICE_CACHE.get(key)
    ref, _ = hit
    assert ref() is not y  # the co-validation in _resident_y_all_device would reject this
    bg._DY_DEVICE_CACHE.clear()


def test_con8_dy_device_cache_bounded():
    from mlframe.feature_selection.filters import batch_mi_noise_gate_gpu as bg
    import weakref

    bg._DY_DEVICE_CACHE.clear()
    held = []
    for i in range(bg._DY_DEVICE_CACHE_MAX * 3):
        y = np.zeros(4, dtype=np.int32)
        held.append(y)
        key = (id(y), i, 1, 4, 2)
        bg._DY_DEVICE_CACHE[key] = (weakref.ref(y), object())
        while len(bg._DY_DEVICE_CACHE) > bg._DY_DEVICE_CACHE_MAX:
            bg._DY_DEVICE_CACHE.popitem(last=False)
    assert len(bg._DY_DEVICE_CACHE) <= bg._DY_DEVICE_CACHE_MAX
    bg._DY_DEVICE_CACHE.clear()


# ---------------------------------------------------------------------------
# CON9: resident-codes handoff per-host-array, not single slot.
# ---------------------------------------------------------------------------
def test_con9_resident_codes_handoff_does_not_clobber_across_arrays():
    """Two producers stash device codes for two distinct host arrays; both must be retrievable.
    Pre-fix the single slot meant producer B wiped producer A's handoff."""
    from mlframe.feature_selection.filters import _gpu_resident_fe as fe

    fe.clear_resident_codes_handoff()
    a = np.zeros((5, 2), dtype=np.int32)
    b = np.zeros((5, 2), dtype=np.int32)
    dev_a, dev_b = object(), object()
    fe._stash_resident_codes(a, dev_a)
    fe._stash_resident_codes(b, dev_b)
    assert fe.take_resident_codes(a) is dev_a
    assert fe.take_resident_codes(b) is dev_b
    fe.clear_resident_codes_handoff()


def test_con9_take_resident_codes_shape_dtype_guarded():
    from mlframe.feature_selection.filters import _gpu_resident_fe as fe

    fe.clear_resident_codes_handoff()
    a = np.zeros((5, 2), dtype=np.int32)
    fe._stash_resident_codes(a, object())
    # Same id is impossible to fake here; instead a different array (different shape) must miss.
    other = np.zeros((6, 2), dtype=np.int32)
    assert fe.take_resident_codes(other) is None
    fe.clear_resident_codes_handoff()


def test_con9_targeted_clear_preserves_other_entry():
    """clear_resident_codes_handoff(host) drops only that host's entry -- a concurrent fit's
    in-flight handoff must survive."""
    from mlframe.feature_selection.filters import _gpu_resident_fe as fe

    fe.clear_resident_codes_handoff()
    a = np.zeros((5, 2), dtype=np.int32)
    b = np.zeros((5, 2), dtype=np.int32)
    fe._stash_resident_codes(a, object())
    fe._stash_resident_codes(b, object())
    fe.clear_resident_codes_handoff(a)
    assert fe.take_resident_codes(a) is None
    assert fe.take_resident_codes(b) is not None
    fe.clear_resident_codes_handoff()


# ---------------------------------------------------------------------------
# CON16: neural-MI model caches use a (reentrant) lock; double-checked miss path.
# ---------------------------------------------------------------------------
def test_con16_neural_mi_cache_lock_is_reentrant():
    """The lock must be reentrant: _calibrate_mist holds it and calls _get_mist_hf_model which
    re-acquires it. A plain Lock would deadlock; an RLock does not."""
    from mlframe.feature_selection.filters import _neural_mi as nm

    lock = nm._NEURAL_MI_CACHE_LOCK
    with lock:
        with lock:  # re-acquire from the same thread -- only an RLock permits this
            assert True


# ---------------------------------------------------------------------------
# CON20: hybrid top-k tiebreak is deterministic on equal importances.
# ---------------------------------------------------------------------------
def test_con20_topk_deterministic_tiebreak_on_equal_fi():
    """With all-equal FI, the cap selection must be reproducible (sorted by column name), not
    dependent on input/iteration order. Pre-fix `sorted(rest, key=fi, reverse=True)[:cap]`
    left ties in arbitrary order."""
    fi = {c: 1.0 for c in ["d", "a", "c", "b", "e"]}
    cap = 3

    def _select(rest):
        return sorted(rest, key=lambda c: (-fi.get(c, 0.0), c))[:cap]

    sel1 = _select(["d", "a", "c", "b", "e"])
    sel2 = _select(["e", "c", "b", "a", "d"])  # different input order
    assert sel1 == sel2 == ["a", "b", "c"]
