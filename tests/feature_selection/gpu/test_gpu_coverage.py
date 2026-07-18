"""Coverage tests for ``mlframe.feature_selection.filters.gpu``.

Exercises the CuPy raw-kernel entry points (``init_kernels``,
``mi_direct_gpu``, ``mi_direct_gpu_batched``, ``mi_direct_gpu_batched_pairs``)
and the persistent device-buffer pool so the line coverage of ``gpu.py``
climbs past 60 percent on a host with a single CUDA device.

The CuPy raw kernel ``compute_mi_from_classes_cuda`` indexes joint counts
as ``joint_counts[i*2 + j]``, which is only numerically correct when
``nbins_y == 2``. CPU-vs-GPU equality checks therefore stay on the
nbins_y=2 regime; larger nbins_y is only used to drive the joint-histogram
kernel through its ``nbins_x * nbins_y`` branch without asserting on the MI
value returned by the legacy mi kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

# Hard requirement -- the GPU file is fully CuPy-driven.
cp = pytest.importorskip("cupy")


# Sanity: this whole file is only meaningful with at least one CUDA device.
def _gpu_available() -> bool:
    """True iff cupy can see at least one CUDA device, gating whether the GPU coverage tests can run at all."""
    try:
        import cupy as _cp

        return _cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:  # pragma: no cover - no driver / no GPU
        return False


_GPU_AVAILABLE = _gpu_available()
if not _GPU_AVAILABLE:  # pragma: no cover - guarded at collection time
    pytest.skip("No CUDA device available", allow_module_level=True)


from mlframe.feature_selection.filters import gpu as gpu_mod
from mlframe.feature_selection.filters.gpu import (
    _GPU_POOL,
    init_kernels,
    mi_direct_gpu,
    mi_direct_gpu_batched,
    mi_direct_gpu_batched_pairs,
)
from mlframe.feature_selection.filters.permutation import mi_direct

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_factors(n=300, nbins_x=5, nbins_y=2, seed=0, dtype=np.int32):
    """Build a small (factors_data, factors_nbins) pair with a real signal.

    ``y`` is derived from ``x + noise`` so MI is reliably > 0 and the
    permutation path explores the failure / success branches.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nbins_x, size=n).astype(dtype)
    if nbins_y == 2:
        y = ((x + rng.normal(scale=0.8, size=n)) > (nbins_x - 1) / 2.0).astype(dtype)
    else:
        # Generic signal: linearly map x noise into nbins_y categories.
        raw = x + rng.normal(scale=0.5, size=n)
        bins = np.linspace(raw.min() - 1e-9, raw.max() + 1e-9, nbins_y + 1)
        y = np.clip(np.digitize(raw, bins) - 1, 0, nbins_y - 1).astype(dtype)
    factors = np.column_stack([x, y]).astype(dtype)
    factors_nbins = np.array([nbins_x, nbins_y], dtype=np.int64)
    return factors, factors_nbins


# ---------------------------------------------------------------------------
# init_kernels & lazy guard
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_init_kernels_idempotent_and_attaches_module_globals():
    """``init_kernels`` is safe to call repeatedly and populates the three
    advertised kernels as ``cp.RawKernel`` objects on the module namespace."""
    import cupy as cp

    init_kernels()
    init_kernels()  # second call must be a no-op without raising.

    assert isinstance(gpu_mod.compute_joint_hist_cuda, cp.RawKernel)
    assert isinstance(gpu_mod.compute_mi_from_classes_cuda, cp.RawKernel)
    assert isinstance(gpu_mod.compute_joint_hist_batched_cuda, cp.RawKernel)
    # The multi-pair kernel is also populated by init_kernels even though it
    # is declared as a fallback ``None`` placeholder below the function body.
    assert isinstance(gpu_mod.compute_joint_hist_multi_pair_cuda, cp.RawKernel)


# ---------------------------------------------------------------------------
# mi_direct_gpu: correctness vs CPU mi_direct on nbins_y == 2
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("input_dtype", [np.int8, np.int16, np.int32])
def test_mi_direct_gpu_matches_cpu_npermutations_zero(input_dtype):
    """With ``npermutations=0`` the GPU path skips its loop entirely and
    just computes ``original_mi`` -- which must match the CPU reference to
    machine precision for every reasonable input integer dtype."""
    factors, factors_nbins = _make_factors(n=400, seed=1, dtype=input_dtype)

    cpu_mi, cpu_conf = mi_direct(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=0,
        parallelism="none",
    )
    gpu_mi, gpu_conf = mi_direct_gpu(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=0,
    )

    assert abs(cpu_mi - gpu_mi) < 1e-6
    # No permutations were run -> confidence is exactly 0.0 from both paths.
    assert cpu_conf == 0.0
    assert gpu_conf == 0.0


@pytest.mark.gpu
def test_mi_direct_gpu_matches_cpu_with_permutations_small_signal():
    """``original_mi`` must agree between CPU and GPU even when the
    permutation loop runs (it's computed once before the loop). Confidence
    is path-dependent because GPU uses a different RNG, so we don't assert
    on it -- only that it lives in ``[0, 1]``."""
    factors, factors_nbins = _make_factors(n=500, nbins_x=5, nbins_y=2, seed=7)

    cpu_mi, _ = mi_direct(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=0,
        parallelism="none",
    )
    gpu_mi, gpu_conf = mi_direct_gpu(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=20,
    )

    # Same closed-form MI; both should round to identical bits because
    # ``compute_mi_from_classes`` is a pure CPU call in both paths for
    # original_mi (the GPU loop only touches the permuted-MI samples).
    assert abs(cpu_mi - gpu_mi) < 1e-6
    assert 0.0 <= gpu_conf <= 1.0


@pytest.mark.gpu
def test_mi_direct_gpu_return_null_mean_contract_and_backcompat():
    """audit5-P1: the GPU relevance path must expose the empirical null mean + p-value so the caller can apply
    the SAME significance-gated debiasing as the CPU branch (it previously returned raw plug-in MI). Pin the
    4-tuple contract (finite null_mean >= 0, p_value in [0,1]) AND that the legacy 2-tuple is unchanged when
    the flag is off."""
    factors, factors_nbins = _make_factors(n=600, nbins_x=6, nbins_y=2, seed=11)

    # back-compat: default returns the legacy 2-tuple.
    legacy = mi_direct_gpu(factors, (0,), (1,), factors_nbins, npermutations=8)
    assert len(legacy) == 2

    res = mi_direct_gpu(factors, (0,), (1,), factors_nbins, npermutations=8, return_null_mean=True)
    assert len(res) == 4
    mi, _conf, null_mean, p_value = res
    assert mi == legacy[0]  # observed MI is identical; only extra outputs are added
    assert np.isfinite(null_mean) and null_mean >= 0.0
    assert 0.0 <= p_value <= 1.0


# ---------------------------------------------------------------------------
# mi_direct_gpu: permutation, cache, dtype-mismatch, large nbins paths
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_mi_direct_gpu_confidence_in_unit_interval_under_shuffle():
    """The permutation path must return confidence in ``[0, 1]`` and the
    pool buffers must reflect the just-run shapes."""
    factors, factors_nbins = _make_factors(n=400, nbins_x=4, nbins_y=2, seed=3)

    original_mi, confidence = mi_direct_gpu(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=30,
    )
    assert 0.0 <= confidence <= 1.0
    # original_mi may be zeroed out by the max_failed early-exit, so we only
    # require that it is finite and non-negative.
    assert np.isfinite(original_mi) and original_mi >= 0.0

    # Pool was populated by the call above.
    assert _GPU_POOL.cap_n >= 400
    assert _GPU_POOL.cap_nbins_x >= 4
    assert _GPU_POOL.cap_nbins_y >= 2


@pytest.mark.gpu
def test_mi_direct_gpu_reuses_pool_across_calls():
    """A second call with the same shape must NOT grow the pool. We assert
    on the underlying CuPy buffer identity to verify the cache hit."""
    factors, factors_nbins = _make_factors(n=400, nbins_x=5, nbins_y=2, seed=11)

    mi_direct_gpu(factors, (0,), (1,), factors_nbins, npermutations=10)
    buf_x_id = id(_GPU_POOL.classes_x)
    buf_y_id = id(_GPU_POOL.classes_y)
    cap_n_before = _GPU_POOL.cap_n

    mi_direct_gpu(factors, (0,), (1,), factors_nbins, npermutations=10)

    assert id(_GPU_POOL.classes_x) == buf_x_id
    assert id(_GPU_POOL.classes_y) == buf_y_id
    assert _GPU_POOL.cap_n == cap_n_before


@pytest.mark.gpu
def test_mi_direct_gpu_accepts_caller_supplied_classes_y_safe():
    """``classes_y_safe`` / ``freqs_y_safe`` override the pool's GPU buffers
    so callers can pre-stage shared work. Pass a freshly-allocated CuPy
    buffer and confirm the call still succeeds."""
    factors, factors_nbins = _make_factors(n=400, nbins_x=5, nbins_y=2, seed=13)

    # Re-derive merged classes_y on host -> GPU explicitly so we exercise
    # the "caller passed GPU arrays" branch.
    from mlframe.feature_selection.filters.info_theory import merge_vars

    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors,
        vars_indices=(1,),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )
    classes_y_gpu = cp.asarray(classes_y.astype(np.int32))
    freqs_y_gpu = cp.asarray(freqs_y.astype(np.float64))

    mi, conf = mi_direct_gpu(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=10,
        classes_y=classes_y,
        freqs_y=freqs_y,
        classes_y_safe=classes_y_gpu,
        freqs_y_safe=freqs_y_gpu,
    )
    assert np.isfinite(mi) and mi >= 0.0
    assert 0.0 <= conf <= 1.0


@pytest.mark.gpu
def test_mi_direct_gpu_handles_large_nbins_x():
    """Drives ``compute_joint_hist_cuda`` with a wider X support so the
    joint-counts buffer grows past the prior cached shape. Only correctness
    of the run (no exception, finite outputs) is asserted -- the legacy
    ``compute_mi_from_classes_cuda`` kernel is hard-wired to nbins_y=2."""
    factors, factors_nbins = _make_factors(
        n=300,
        nbins_x=16,
        nbins_y=2,
        seed=17,
    )

    mi, conf = mi_direct_gpu(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=15,
    )
    assert np.isfinite(mi) and mi >= 0.0
    assert 0.0 <= conf <= 1.0


@pytest.mark.gpu
def test_mi_direct_gpu_handles_zero_signal_short_circuit():
    """When ``original_mi == 0`` the GPU path returns immediately without
    permuting. Build a synthetic dataset where x and y are independent
    enough that the integer-bin MI lands at exactly 0 or near zero."""
    rng = np.random.default_rng(0)
    n = 50  # tiny n so binomial bins are very likely to flatten.
    x = np.zeros(n, dtype=np.int32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    factors = np.column_stack([x, y])
    factors_nbins = np.array([1, 2], dtype=np.int64)

    mi, conf = mi_direct_gpu(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=20,
    )
    # x is a constant column -> MI is identically zero.
    assert mi == 0.0
    assert conf == 0.0


# ---------------------------------------------------------------------------
# mi_direct_gpu_batched: batch loop + OOM clamp
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_mi_direct_gpu_batched_returns_valid_tuple():
    """Smoke test: ``mi_direct_gpu_batched`` returns ``(mi, confidence)``
    with finite values for a small permutation budget."""
    factors, factors_nbins = _make_factors(n=400, nbins_x=5, nbins_y=2, seed=21)

    mi, conf = mi_direct_gpu_batched(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=40,
        batch_size=8,
    )
    assert np.isfinite(mi) and mi >= 0.0
    assert 0.0 <= conf <= 1.0


@pytest.mark.gpu
def test_mi_direct_gpu_batched_zero_perms_returns_original_mi():
    """``npermutations=0`` skips the loop and yields the closed-form MI
    plus zero confidence -- mirrors the ``mi_direct_gpu`` contract."""
    factors, factors_nbins = _make_factors(n=300, nbins_x=5, nbins_y=2, seed=22)

    cpu_mi, _ = mi_direct(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=0,
        parallelism="none",
    )
    gpu_mi, gpu_conf = mi_direct_gpu_batched(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=0,
        batch_size=8,
    )
    assert abs(cpu_mi - gpu_mi) < 1e-6
    assert gpu_conf == 0.0


@pytest.mark.gpu
def test_mi_direct_gpu_batched_accepts_cached_classes_y():
    """The ``classes_y`` / ``freqs_y`` keyword path lets callers reuse
    pre-merged label arrays across multiple calls."""
    factors, factors_nbins = _make_factors(n=400, nbins_x=5, nbins_y=2, seed=23)
    from mlframe.feature_selection.filters.info_theory import merge_vars

    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors,
        vars_indices=(1,),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )
    mi, conf = mi_direct_gpu_batched(
        factors,
        (0,),
        (1,),
        factors_nbins,
        npermutations=20,
        batch_size=8,
        classes_y=classes_y,
        freqs_y=freqs_y,
    )
    assert np.isfinite(mi) and mi >= 0.0
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# mi_direct_gpu_batched_pairs: multi-pair joint MI
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_mi_direct_gpu_batched_pairs_returns_aligned_float64_vector():
    """One pair per column-combination, output is ``float64`` 1-D of length
    ``len(pairs_a)`` and every entry is >= 0 (MI is non-negative)."""
    rng = np.random.default_rng(31)
    n = 300
    n_cols = 4
    factors_data = rng.integers(0, 3, size=(n, n_cols)).astype(np.int32)
    factors_nbins = np.array([3] * n_cols, dtype=np.int64)
    classes_y = rng.integers(0, 2, size=n).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64)
    freqs_y /= freqs_y.sum()

    pairs_a = np.array([0, 0, 1], dtype=np.int32)
    pairs_b = np.array([1, 2, 2], dtype=np.int32)

    out = mi_direct_gpu_batched_pairs(
        factors_data=factors_data,
        pairs_a=pairs_a,
        pairs_b=pairs_b,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
    )
    assert out.dtype == np.float64
    assert out.shape == (3,)
    assert np.all(out >= 0)


@pytest.mark.gpu
def test_mi_direct_gpu_batched_pairs_empty_returns_empty_array():
    """Zero-pair edge case returns an empty float64 array without launching
    a kernel."""
    factors_data = np.zeros((10, 2), dtype=np.int32)
    factors_nbins = np.array([2, 2], dtype=np.int64)
    classes_y = np.zeros(10, dtype=np.int32)
    freqs_y = np.array([1.0], dtype=np.float64)

    out = mi_direct_gpu_batched_pairs(
        factors_data=factors_data,
        pairs_a=np.array([], dtype=np.int32),
        pairs_b=np.array([], dtype=np.int32),
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
    )
    assert out.dtype == np.float64
    assert out.shape == (0,)


@pytest.mark.gpu
def test_mi_direct_gpu_batched_pairs_memory_guard_raises_on_overlarge_request():
    """The ``total_cells * 4 > 4 GiB`` guard must raise ``MemoryError``
    before any kernel launch is attempted."""
    # The function eagerly imports cupy on the first line; on no-GPU hosts
    # that import raises a wrapped RuntimeError, not MemoryError. The
    # MemoryError-guard contract is only exercisable when cupy is importable.
    pytest.importorskip("cupy")
    # nbins_a=nbins_b=2048 -> merged 2048*2048 = ~4.2M cells per pair;
    # times nbins_y=128 = ~537M cells; times 4 bytes = 2.15 GiB -- below
    # the 4 GiB ceiling. Use two pairs so we cross the limit.
    n = 4
    n_cols = 2
    factors_data = np.zeros((n, n_cols), dtype=np.int32)
    factors_nbins = np.array([2048, 2048], dtype=np.int64)
    classes_y = np.zeros(n, dtype=np.int32)
    freqs_y = np.ones(128, dtype=np.float64) / 128.0
    pairs_a = np.array([0, 0], dtype=np.int32)
    pairs_b = np.array([1, 1], dtype=np.int32)

    with pytest.raises(MemoryError):
        mi_direct_gpu_batched_pairs(
            factors_data=factors_data,
            pairs_a=pairs_a,
            pairs_b=pairs_b,
            factors_nbins=factors_nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
        )


# ---------------------------------------------------------------------------
# Internal: _GpuBufferPool grow semantics
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_buffer_pool_grows_monotonically():
    """The pool ``ensure()`` method must enlarge each cached buffer only
    when the requested capacity exceeds the current capacity."""
    from mlframe.feature_selection.filters.gpu import _GpuBufferPool

    pool = _GpuBufferPool()
    pool.ensure(n=128, nbins_x=4, nbins_y=2)
    assert pool.cap_n == 128
    assert pool.cap_nbins_x == 4
    assert pool.cap_nbins_y == 2

    cx0 = pool.classes_x
    fx0 = pool.freqs_x
    # Re-request a SMALLER size -- pool must keep the larger buffers in place.
    pool.ensure(n=64, nbins_x=2, nbins_y=2)
    assert pool.cap_n == 128
    assert pool.classes_x is cx0
    assert pool.freqs_x is fx0

    # Grow along the n axis only.
    pool.ensure(n=256, nbins_x=4, nbins_y=2)
    assert pool.cap_n == 256
    assert pool.classes_x is not cx0
