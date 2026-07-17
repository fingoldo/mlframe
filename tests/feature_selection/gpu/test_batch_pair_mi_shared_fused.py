"""Regression tests for the single-launch, opt-in-dynamic-shared-memory pair-MI kernel
(``_batch_pair_mi_cuda_shared_fused.py``) and its two accompanying fixes.

Root cause (2026-07-16, wellbore-100k cProfile on a quiet/uncontended machine): the production shape
(n_classes_y=20, max_joint up to 441-528) exceeds ``batch_pair_mi_cuda``'s STATIC shared-memory kernel
caps (``MAX_JOINT_BINS_CUDA=256`` / ``MAX_Y_BINS_CUDA=16``, sized for the 48KB-per-block budget every
CUDA device guarantees WITHOUT opt-in), forcing every call onto ``batch_pair_mi_cuda_row_chunked``.
That kernel's own histogram computation is already efficient, but its outer row/pair-chunking loop
fragments into many small launches under VRAM pressure (151+ launches measured at 200MB free for an
85k-pair production call, vs a theoretical 3 at 3GB free) -- 78-92s of a ~500-585s fit wall in the
profile. Since ``factors_data`` at this scale (~206MB) trivially fits VRAM whole, chunking is
unnecessary; the real blocker was only the STATIC shared-memory cap.

``batch_pair_mi_cuda_shared_fused`` removes it via a CuPy ``RawKernel`` (``cp.RawKernel`` -- not
``numba.cuda.jit``, which offers no supported way to opt into >48KB dynamic shared memory in this numba
version) using ``extern __shared__`` DYNAMIC shared memory sized at RUNTIME, with
``max_dynamic_shared_size_bytes`` opting into the device's EXTENDED per-block budget (verified live:
48KB static vs 99KB opt-in on this host's RTX 500 Ada, cc 8.9). One launch, one block per pair, no
row/pair chunking, no global accumulator.

Two bugs were found and fixed en route (both regression-tested here):

1. An early version of the fused kernel reduced the joint histogram to the final MI scalar on a SINGLE
   thread per block (``if (tid==0) { for(...) for(...) mi += ...log(...) }``), leaving the block's other
   127 threads idle during that phase -- measured ~6x slower end-to-end than the accumulate-only phase
   alone would suggest (isolated A/B: 3.3s vs 0.45s at n_pairs=20000, same total FLOPs). Fixed by
   parallelizing the reduction across the block via a shared ``atomicAdd`` accumulator (every thread
   strides over a disjoint subset of joint cells).
2. ``batch_pair_mi_cupy`` (the pre-existing plain-CuPy backend, untouched by the fused-kernel work but
   discovered via this suite's regression coverage) never got the ``resident_operand`` upload-dedup fix
   that ``batch_pair_mi_cuda`` received on 2026-07-12 -- it re-uploaded ``factors_data`` via a raw
   ``cp.asarray`` on every call. Fixed by routing it through the same content-hash-keyed cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
    _CUDA_AVAIL,
    _CUPY_AVAIL,
    batch_pair_mi_cupy,
    batch_pair_mi_njit_prange,
    dispatch_batch_pair_mi,
)
from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import (
    batch_pair_mi_cuda_shared_fused,
    shared_fused_kernel_fits_budget,
)
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

pytestmark = pytest.mark.skipif(not (_CUDA_AVAIL and _CUPY_AVAIL), reason="numba.cuda + cupy required for this suite")


@pytest.fixture(autouse=True)
def _clean_gpu_state():
    """Clear resident FE GPU operands before and after every test so cached uploads never leak across tests."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _build_pair_inputs(n_samples, n_features, n_pairs, n_classes_y, nbins_range, seed):
    """Build a random factors/pair/class fixture for the shared-fused-kernel test suite."""
    rng = np.random.default_rng(seed)
    nbins = rng.integers(nbins_range[0], nbins_range[1] + 1, n_features).astype(np.int32)
    factors_data = np.column_stack([rng.integers(0, int(nb), n_samples) for nb in nbins]).astype(np.int32)
    classes_y = rng.integers(0, n_classes_y, n_samples).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=n_classes_y).astype(np.float64) / n_samples
    pair_a = rng.integers(0, n_features, n_pairs).astype(np.int64)
    pair_b = ((pair_a + rng.integers(1, n_features, n_pairs)) % n_features).astype(np.int64)
    return factors_data, nbins, classes_y, freqs_y, pair_a, pair_b


class TestBitIdentity:
    """The kernel must match the CPU njit reference within the established ~1e-9 FP-reorder tolerance
    (atomicAdd accumulation across threads reorders the summation vs the CPU's serial i-then-j loop)."""

    @pytest.mark.parametrize(
        "n_samples,n_features,n_pairs,n_classes_y,nbins_range,seed",
        [
            (99401, 30, 200, 20, (2, 25), 1),  # production-like: n_classes_y=20, wide nbins
            (500, 10, 30, 3, (2, 5), 2),  # small
            (2000, 8, 20, 5, (2, 3), 3),  # low-cardinality / tie-heavy
            (20000, 15, 50, 20, (15, 21), 4),  # large max_joint near the static cap (441)
        ],
    )
    def test_matches_cpu_reference(self, n_samples, n_features, n_pairs, n_classes_y, nbins_range, seed):
        """The shared-fused GPU kernel matches the CPU njit reference within a ~1e-9 FP-reorder tolerance."""
        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
            n_samples,
            n_features,
            n_pairs,
            n_classes_y,
            nbins_range,
            seed,
        )
        max_joint = int((nbins[pair_a].astype(np.int64) * nbins[pair_b].astype(np.int64)).max())
        if shared_fused_kernel_fits_budget(max_joint, n_classes_y) == 0:
            pytest.skip(f"max_joint={max_joint} n_classes_y={n_classes_y} exceeds this device's opt-in shared-memory budget")
        mi_gpu = batch_pair_mi_cuda_shared_fused(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        mi_cpu = batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        np.testing.assert_allclose(mi_gpu, mi_cpu, atol=1e-9, rtol=1e-9)


class TestBudgetGate:
    """``shared_fused_kernel_fits_budget`` must reject shapes that would overflow even the opt-in budget,
    never silently truncate or launch an oversized allocation."""

    def test_tiny_shape_fits(self):
        """A tiny (max_joint, n_classes_y) shape fits the opt-in shared-memory budget."""
        assert shared_fused_kernel_fits_budget(4, 2) > 0

    def test_absurdly_large_shape_rejected(self):
        """A shape far exceeding any real device's shared-memory budget is rejected (returns 0)."""
        # 100_000 * 10_000 int32 cells -> far beyond any real device's opt-in shared-memory budget.
        assert shared_fused_kernel_fits_budget(100_000, 10_000) == 0

    def test_rejected_shape_raises_cleanly_not_silently_wrong(self):
        """Calling the kernel directly on a budget-rejected shape raises RuntimeError rather than silently misbehaving."""
        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(200, 3, 3, 5000, (2, 3), 7)
        with pytest.raises(RuntimeError):
            batch_pair_mi_cuda_shared_fused(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)


class TestDispatcherIntegration:
    """dispatch_batch_pair_mi must route to the shared-fused backend when the static-shared kernel's
    shape guard trips but the shape fits the opt-in budget -- the exact fallback chain this kernel exists
    to shortcut (static kernel -> shared-fused -> row-chunked -> CPU)."""

    def test_forced_cuda_routes_to_shared_fused_for_wide_y(self):
        """dispatch_batch_pair_mi routes to cuda_shared_fused when n_classes_y exceeds the static kernel's cap."""
        # n_classes_y=20 > MAX_Y_BINS_CUDA=16 -> batch_pair_mi_cuda raises -> shared-fused should catch it.
        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(99401, 30, 200, 20, (15, 22), 8)
        mi, backend = dispatch_batch_pair_mi(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y, force_backend="cuda")
        assert backend == "cuda_shared_fused"
        mi_cpu = batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        np.testing.assert_allclose(mi, mi_cpu, atol=1e-9, rtol=1e-9)

    def test_small_shape_still_uses_static_kernel(self):
        """dispatch_batch_pair_mi still routes a shape that fits the static kernel's caps to the static kernel."""
        # A shape that fits the STATIC kernel's caps must still use it -- the new backend must not
        # preempt the (equally correct, no dynamic-shared-memory dependency) faster default path.
        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(2000, 6, 10, 4, (2, 4), 9)
        _mi, backend = dispatch_batch_pair_mi(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y, force_backend="cuda")
        assert backend == "cuda"


class TestParallelReductionRegression:
    """Regression sensor for the single-thread-serial-reduction bug: an early version of this kernel
    computed the whole (max_joint, n_classes_y) MI reduction on ONE thread per block (``if tid==0``),
    measured ~6x slower than the parallelized (atomicAdd-per-thread) version at equal total work.
    A tight wall-clock ceiling well above the parallel version's real cost but well below the serial
    version's catches a regression back to the serial form without being a flaky microbenchmark."""

    def test_reduction_is_parallelized_not_serial(self):
        """A warm shared-fused-kernel call completes well under the serial-reduction regression ceiling."""
        import time

        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(50000, 20, 5000, 20, (15, 22), 10)
        # warm (JIT/NVRTC compile + first launch)
        batch_pair_mi_cuda_shared_fused(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        t0 = time.perf_counter()
        batch_pair_mi_cuda_shared_fused(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        elapsed = time.perf_counter() - t0
        # Parallel version measured ~0.1-0.5s at this shape on the reference host; serial measured ~3s+
        # at a comparable shape. 2.0s leaves generous margin for slower hardware while still catching a
        # regression back to single-thread reduction.
        assert elapsed < 2.0, f"shared-fused kernel took {elapsed:.2f}s -- possible regression to a serial (non-parallelized) reduction"


class TestCupyResidentUploadRegression:
    """Regression sensor for the missed resident_operand fix on batch_pair_mi_cupy: two calls with the
    SAME (factors_data, classes_y, freqs_y) content but DIFFERENT pair chunks (mirrors two pair-
    subchunks of one greedy round) must upload factors_data via cp.asarray only ONCE. Fails pre-fix
    (n=2) and passes post-fix (n=1) -- verified empirically before shipping."""

    def test_batch_pair_mi_cupy_uploads_factors_data_once_across_calls(self):
        """Two batch_pair_mi_cupy calls sharing identical factors_data content upload it via cp.asarray only once."""
        import cupy as cp

        factors_data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(2000, 6, 10, 4, (2, 5), 11)
        half = len(pair_a) // 2
        assert half > 0

        upload_calls = {"n": 0}
        orig_asarray = cp.asarray

        def _counting_asarray(arr, *a, **kw):
            """Count cp.asarray calls whose input shape matches factors_data (a proxy for redundant uploads)."""
            if getattr(arr, "shape", None) == factors_data.shape:
                upload_calls["n"] += 1
            return orig_asarray(arr, *a, **kw)

        cp.asarray = _counting_asarray
        try:
            mi1 = batch_pair_mi_cupy(factors_data, pair_a[:half], pair_b[:half], nbins, classes_y, freqs_y)
            mi2 = batch_pair_mi_cupy(factors_data, pair_a[half:], pair_b[half:], nbins, classes_y, freqs_y)
        finally:
            cp.asarray = orig_asarray

        assert upload_calls["n"] == 1, f"factors_data-shaped cp.asarray called {upload_calls['n']} times across 2 calls (expected 1)"

        mi_cpu = batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
        np.testing.assert_allclose(np.concatenate([mi1, mi2]), mi_cpu, atol=1e-9, rtol=1e-9)
