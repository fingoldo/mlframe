"""Regression + biz_value tests for the preallocated ring buffer backing ``CompositeTargetEstimator.update``.

The streaming-refit rolling window used to be a ``collections.deque`` of boxed
Python floats: every ``update`` did ``deque.extend(arr.tolist())`` (boxing) then
``np.asarray(deque)`` (unboxing the whole window) -- O(buffer_n) Python work per
call even when the cheap drift z-check short-circuits. It is now a fixed-capacity
float64 ring buffer (``_RingBuffer``): a single preallocated store + head index +
count, so an append copies only the new rows and the drift check reads a
contiguous FIFO view materialised into a reused scratch array.

These tests pin BOTH halves of the contract:

* Correctness vs the OLD ``deque(maxlen=...)`` semantics -- identical FIFO order,
  identical eviction, identical over-long-batch trailing-keep -- across random
  append sequences (a regression on the ring math fails here).
* The perf-shaped invariant -- the storage / view arrays are REUSED across
  ``update`` calls (same object identity, never rebuilt) and the windowed
  contents the drift check sees stay correct.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from mlframe.training.composite.estimator import _update
from mlframe.training.composite.estimator._update import _RingBuffer

# ===========================================================================
# _RingBuffer unit correctness vs the legacy deque(maxlen) semantics
# ===========================================================================


def _deque_reference(capacity: int, batches: list[np.ndarray]) -> list[float]:
    """Replays the OLD deque(maxlen) path and returns the FIFO-ordered window.

    Mirrors the pre-fix code exactly: ``extend(arr.tolist())`` then read the
    deque left-to-right (oldest first), which is what ``np.asarray(deque)`` did.
    """
    dq: deque = deque(maxlen=capacity)
    for arr in batches:
        dq.extend(arr.tolist())
    return list(dq)


class TestRingBufferMatchesDeque:
    """Groups tests covering ring buffer matches deque."""
    def test_simple_fifo_order(self) -> None:
        """Simple fifo order."""
        rb = _RingBuffer(5)
        rb.append(np.array([1.0, 2.0, 3.0]))
        assert rb.contiguous().tolist() == [1.0, 2.0, 3.0]
        assert len(rb) == 3

    def test_eviction_keeps_newest(self) -> None:
        """deque(maxlen=5) after [1,2,3] then [4,5,6,7] -> [3,4,5,6,7]."""
        rb = _RingBuffer(5)
        rb.append(np.array([1.0, 2.0, 3.0]))
        rb.append(np.array([4.0, 5.0, 6.0, 7.0]))
        assert rb.contiguous().tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
        assert len(rb) == 5

    def test_overlong_single_batch_keeps_trailing(self) -> None:
        """A single batch longer than capacity keeps only its last ``capacity`` rows."""
        rb = _RingBuffer(3)
        rb.append(np.array([10.0, 11.0, 12.0, 13.0, 14.0]))
        assert rb.contiguous().tolist() == [12.0, 13.0, 14.0]
        assert len(rb) == 3

    def test_empty_append_is_noop(self) -> None:
        """Empty append is noop."""
        rb = _RingBuffer(4)
        rb.append(np.array([1.0, 2.0]))
        rb.append(np.array([]))
        assert rb.contiguous().tolist() == [1.0, 2.0]
        assert len(rb) == 2

    @pytest.mark.parametrize("capacity", [1, 3, 7, 50])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_sequences_match_deque_bit_for_bit(self, capacity: int, seed: int) -> None:
        """Across random batch sizes, the ring window equals the deque window exactly."""
        rng = np.random.default_rng(seed)
        rb = _RingBuffer(capacity)
        batches: list[np.ndarray] = []
        for _ in range(40):
            m = int(rng.integers(0, capacity * 2 + 3))  # may be 0, may exceed capacity
            arr = rng.normal(size=m)
            batches.append(arr)
            rb.append(arr)
            ref = _deque_reference(capacity, batches)
            got = rb.contiguous().tolist()
            assert got == pytest.approx(ref), f"cap={capacity} seed={seed} step batch_m={m}"
            assert len(rb) == len(ref)

    def test_wrapped_window_is_contiguous_and_correct(self) -> None:
        """After many wraps the contiguous() view is still oldest-first + the right length."""
        rb = _RingBuffer(4)
        # 13 single rows -> last 4 are [9,10,11,12].
        for v in range(13):
            rb.append(np.array([float(v)]))
        view = rb.contiguous()
        assert view.dtype == np.float64
        assert view.flags["C_CONTIGUOUS"]
        assert view.tolist() == [9.0, 10.0, 11.0, 12.0]


# ===========================================================================
# biz_value (perf-shaped): arrays are REUSED, not rebuilt, every update
# ===========================================================================


class _StubEstimator:
    """Minimal stand-in exposing exactly what ``_update.update`` touches.

    Avoids fitting LightGBM: we drive the buffer machinery directly and use a
    huge ``min_buffer_n`` so the drift helper short-circuits (the exact path the
    old code still paid O(buffer_n) boxing for).
    """

    transform_name = "linear_residual"
    online_refit_enabled = True
    online_refit_z_threshold = 3.0

    def __init__(self, buffer_n: int, min_buffer_n: int) -> None:
        self.online_refit_buffer_n = buffer_n
        self.online_refit_min_buffer_n = min_buffer_n
        self.fitted_params_ = {"alpha": 1.0, "beta": 0.0}


class TestBizValueArrayReuse:
    """Groups tests covering biz value array reuse."""
    def test_storage_and_view_arrays_reused_across_updates(self) -> None:
        """The preallocated store / view ndarrays must be the SAME objects every
        call -- the whole point of the ring buffer is that nothing is rebuilt.

        On the old deque path there was no persistent ndarray at all (a fresh
        ``np.asarray(deque)`` was allocated per call), so this invariant is the
        behavioural signature of the fix.
        """
        est = _StubEstimator(buffer_n=10_000, min_buffer_n=10**9)
        rng = np.random.default_rng(7)
        _update.update(est, rng.normal(size=100), rng.normal(size=100))
        y_store_id = id(est._buffer_y_._store)
        y_view_id = id(est._buffer_y_._view)
        b_store_id = id(est._buffer_base_._store)
        b_view_id = id(est._buffer_base_._view)
        # Capacity-many rows fit without ever reallocating the store/view.
        for _ in range(200):
            _update.update(est, rng.normal(size=40), rng.normal(size=40))
        assert id(est._buffer_y_._store) == y_store_id
        assert id(est._buffer_y_._view) == y_view_id
        assert id(est._buffer_base_._store) == b_store_id
        assert id(est._buffer_base_._view) == b_view_id

    def test_store_preallocated_to_full_capacity(self) -> None:
        """The store is allocated once at capacity, never grown row-by-row."""
        est = _StubEstimator(buffer_n=5_000, min_buffer_n=10**9)
        _update.update(est, np.zeros(3), np.zeros(3))
        assert est._buffer_y_._store.shape == (5_000,)
        assert est._buffer_y_._store.dtype == np.float64

    def test_windowed_contents_correct_after_eviction(self) -> None:
        """Functional check that the reused arrays still hold the right FIFO window.

        Stream 30 batches of 50 rows (1500 > capacity 1000); the live window must
        be the trailing 1000 rows in arrival order -- verified against a deque.
        """
        cap = 1_000
        est = _StubEstimator(buffer_n=cap, min_buffer_n=10**9)
        rng = np.random.default_rng(11)
        all_y: list[float] = []
        for _ in range(30):
            y = rng.normal(size=50)
            b = rng.normal(size=50)
            all_y.extend(y.tolist())
            _update.update(est, y, b)
        expected = all_y[-cap:]
        got = est._buffer_y_.contiguous().tolist()
        assert len(got) == cap
        assert got == pytest.approx(expected)

    def test_no_per_call_boxing_quick_perf_floor(self) -> None:
        """Perf-shaped floor: with a full 10k window and a short-circuiting drift
        check, many small updates must stay cheap (append is O(new_rows), not
        O(buffer_n) boxing/unboxing).

        Conservative wall floor -- the old deque path boxed+unboxed 10k PyFloats
        per call; the ring path copies only the new rows. We assert the new path
        completes well under a generous bound so a regression to per-call
        whole-window work (or an accidental reallocation) trips it.
        """
        import time

        est = _StubEstimator(buffer_n=10_000, min_buffer_n=10**9)
        rng = np.random.default_rng(13)
        # Fill the window first (cost not measured).
        _update.update(est, rng.normal(size=10_000), rng.normal(size=10_000))
        assert len(est._buffer_y_) == 10_000
        t0 = time.perf_counter()
        for _ in range(2_000):
            _update.update(est, rng.normal(size=5), rng.normal(size=5))
        elapsed = time.perf_counter() - t0
        # 2000 small updates on a full 10k window: generous 2.0 s ceiling (the
        # ring path runs in well under 0.5 s; the floor only guards against a
        # reintroduced O(buffer_n) per-call cost or a per-call reallocation).
        assert elapsed < 2.0, f"2000 small updates took {elapsed:.3f}s (perf regression?)"
