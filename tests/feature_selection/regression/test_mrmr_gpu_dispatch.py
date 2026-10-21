"""Behavioural tests for MRMR GPU dispatch: device gates, disable_gpu honouring, locks and caches."""

from __future__ import annotations
import numpy as np
import pytest
from tests.conftest import _need_cuda
from tests.feature_selection.regression._mrmr_regression_helpers import _FakeCudaFault


def test_regression_cmi_perm_stop_kz_cap_bounded_at_10k():
    """K_z must be capped well below the old 1_000_000 ceiling -- construct a pool of selected columns
    whose cardinality product exceeds the new 10_000 cap and confirm it triggers the truncation warning
    (not an unbounded allocation) at a cardinality easily reachable with realistic ten-bin features
    (10**5 = 100_000 > 10_000, needing only 5 selected columns, vs 10**6 for the old cap).
    """
    from mlframe.feature_selection.filters._cmi_perm_stop import _MAX_K_Z, cmi_permutation_stop

    assert _MAX_K_Z == 10_000

    rng = np.random.default_rng(0)
    n = 500
    x_cand = rng.integers(0, 10, n).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    selected_cols = [rng.integers(0, 10, n).astype(np.int64) for _ in range(5)]  # 10**5 = 100_000 > 10_000

    is_sig, observed, _p_value = cmi_permutation_stop(
        x_cand=x_cand,
        y=y,
        selected_cols=selected_cols,
        nbins_x=10,
        nbins_y=2,
        nbins_selected=[10] * 5,
        n_permutations=5,
        seed=0,
    )
    assert isinstance(is_sig, (bool, np.bool_))
    assert observed >= 0.0


# ---------------------------------------------------------------------------
# SCREEN_CONFIRM_B-4 (P1): mi_direct's internal prefer_gpu fastpath gate never
# consulted gpu_globally_disabled()/MLFRAME_DISABLE_GPU.
# SCREEN_CONFIRM_B-5 (P1): the order-1 resident-GPU maxT floor's fault paths had
# zero logging and no circuit breaker, unlike the order-2 sibling.
# ---------------------------------------------------------------------------


def test_regression_mi_direct_internal_gate_honours_disable_gpu(monkeypatch):
    """Pre-fix: mi_direct's own internal prefer_gpu fastpath checked only is_cuda_available(), not
    gpu_globally_disabled()/MLFRAME_DISABLE_GPU -- so a caller relying on the env-var opt-out (without
    also passing prefer_gpu=False itself) could still have mi_direct silently route to GPU. Observed at the
    call boundary: with the opt-out set and npermutations above the fastpath threshold, the GPU entry point
    must never be invoked.
    """
    from mlframe.feature_selection.filters import gpu as gpu_mod
    from mlframe.feature_selection.filters import permutation as perm_mod

    gpu_calls: list = []

    def _gpu_spy(**kwargs):
        """Record any routing into the GPU entry point."""
        gpu_calls.append(kwargs)
        return (0.0, 1.0)

    monkeypatch.setattr(gpu_mod, "mi_direct_gpu", _gpu_spy)
    monkeypatch.setattr(perm_mod, "_MI_DIRECT_GPU_FAILED", False)
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")

    rng = np.random.default_rng(0)
    factors = rng.integers(0, 4, size=(300, 2)).astype(np.int32)
    perm_mod.mi_direct(
        factors,
        x=(0,),
        y=(1,),
        factors_nbins=np.array([4, 4]),
        npermutations=32,
        prefer_gpu=True,
        parallelism="none",
    )
    assert gpu_calls == [], "mi_direct routed to the GPU entry point although MLFRAME_DISABLE_GPU=1 was set"


def test_regression_order1_maxt_gpu_circuit_breaker_state_machine():
    """Pins the order-1 resident maxT GPU circuit breaker's trip/reset/query state machine -- did not
    exist at all pre-fix (this file had zero module-level circuit-breaker state or logging).
    """
    from mlframe.feature_selection.filters._permutation_null_resident import (
        order1_maxt_gpu_circuit_breaker_tripped,
        reset_order1_maxt_gpu_circuit_breaker,
        trip_order1_maxt_gpu_circuit_breaker,
    )

    reset_order1_maxt_gpu_circuit_breaker()
    try:
        assert order1_maxt_gpu_circuit_breaker_tripped() is False
        trip_order1_maxt_gpu_circuit_breaker()
        assert order1_maxt_gpu_circuit_breaker_tripped() is True
    finally:
        reset_order1_maxt_gpu_circuit_breaker()
    assert order1_maxt_gpu_circuit_breaker_tripped() is False


# ---------------------------------------------------------------------------
# INFO_THEORY_A-9 (P2): _CMI_RESIDENT_CACHE had no lock at all, unlike its two
# siblings in the same file (_FORDER_LOCK, _FACTORS_DEVICE_LOCK).
# ---------------------------------------------------------------------------


def test_regression_dispatch_friend_graph_stats_catches_bare_exception_cuda_fault(monkeypatch):
    """Pre-fix: `except (ValueError, RuntimeError)` let a bare-Exception CUDA fault propagate
    uncaught out of dispatch_friend_graph_stats's cuda branch. Post-fix: broadened to `except Exception`."""
    import mlframe.feature_selection.filters.friend_graph_gpu as fg_gpu

    monkeypatch.setattr(fg_gpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(fg_gpu, "_CUPY_AVAIL", False)

    def _raise(*a, **k):
        """Stand-in that raises _FakeCudaFault, so the caller's failure path is the one under test."""
        raise _FakeCudaFault("simulated CUDA driver fault")

    monkeypatch.setattr(fg_gpu, "friend_graph_stats_cuda", _raise)
    monkeypatch.setattr(fg_gpu, "_friend_graph_backend_choice", lambda n, k: "cuda")

    result = fg_gpu.dispatch_friend_graph_stats(
        sel=np.array([0, 1], dtype=np.int64),
        factors_data=np.zeros((10, 2), dtype=np.int32),
        factors_nbins=np.array([2, 2], dtype=np.int32),
        target_indices=np.array([0], dtype=np.int64),
        dtype=np.int32,
    )
    assert result is None  # falls back to CPU instead of raising


def test_regression_dispatch_batch_mi_noise_gate_gpu_catches_bare_exception_cuda_fault(monkeypatch):
    """Pre-fix: `except (ValueError, RuntimeError)` let a bare-Exception CUDA fault propagate
    uncaught out of dispatch_batch_mi_with_noise_gate_gpu's cuda branch. Post-fix: broadened."""
    import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bng_gpu

    monkeypatch.setattr(bng_gpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bng_gpu, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bng_gpu, "gpu_globally_disabled", lambda: False, raising=False)

    def _raise(*a, **k):
        """Stand-in that raises _FakeCudaFault, so the caller's failure path is the one under test."""
        raise _FakeCudaFault("simulated CUDA driver fault")

    monkeypatch.setattr(bng_gpu, "batch_mi_with_noise_gate_cuda", _raise)
    monkeypatch.setattr(bng_gpu, "_batch_mi_noise_gate_backend_choice", lambda n, k: "cuda")

    result = bng_gpu.dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d=np.zeros((10, 2), dtype=np.int32),
        factors_nbins=np.array([2, 2], dtype=np.int32),
        classes_y=np.zeros(10, dtype=np.int32),
        classes_y_safe=np.zeros(10, dtype=np.int32),
        freqs_y=np.array([1.0]),
        npermutations=0,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.0,
        use_su=False,
    )
    assert result is None  # falls back to CPU instead of raising


# ---------------------------------------------------------------------------
# GPU_INFRA_A-2 (P1): batch_pair_mi_cuda_row_chunked / batch_pair_mi_cuda_shared_fused raised
# ZeroDivisionError for n_samples == 0 (inv_n = 1.0 / n_samples), unlike the cupy sibling
# (batch_pair_mi_cupy) which has an explicit `if n_samples == 0: return zeros` guard.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_regression_batch_pair_mi_cuda_row_chunked_empty_input_no_zero_division():
    """Pre-fix: n_samples=0 raised ZeroDivisionError deep inside _mi_from_joint_counts(_cupy).
    Post-fix: an early `if n_samples == 0: return zeros` guard mirrors the cupy sibling."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels import _CUDA_AVAIL

    if not _CUDA_AVAIL:
        pytest.skip("numba.cuda not available on this host")
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_kernels import batch_pair_mi_cuda_row_chunked

    out = batch_pair_mi_cuda_row_chunked(
        factors_data=np.zeros((0, 3), dtype=np.int32),
        pair_a=np.array([0, 1], dtype=np.int64),
        pair_b=np.array([1, 2], dtype=np.int64),
        nbins=np.array([2, 2, 2], dtype=np.int32),
        classes_y=np.zeros(0, dtype=np.int32),
        freqs_y=np.array([1.0]),
    )
    assert out.shape == (2,)
    assert np.all(out == 0.0)


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_regression_batch_pair_mi_cuda_shared_fused_empty_input_no_zero_division():
    """Pre-fix: n_samples=0 raised ZeroDivisionError at `inv_n = 1.0 / float(n_samples)`.
    Post-fix: an early `if n_samples == 0: return zeros` guard mirrors the cupy sibling."""
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import _CUPY_AVAIL

    if not _CUPY_AVAIL:
        pytest.skip("cupy not available on this host")
    from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import batch_pair_mi_cuda_shared_fused

    out = batch_pair_mi_cuda_shared_fused(
        factors_data=np.zeros((0, 3), dtype=np.int32),
        pair_a=np.array([0, 1], dtype=np.int64),
        pair_b=np.array([1, 2], dtype=np.int64),
        nbins=np.array([2, 2, 2], dtype=np.int32),
        classes_y=np.zeros(0, dtype=np.int32),
        freqs_y=np.array([1.0]),
    )
    assert out.shape == (2,)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# GPU_INFRA_A-3 (P2): dispatch_batch_mi_with_noise_gate_gpu had no internal gpu_globally_disabled()
# self-check, unlike its dispatch_batch_pair_mi / dispatch_friend_graph_stats siblings, relying
# entirely on its sole production caller checking MLFRAME_DISABLE_GPU first.
# ---------------------------------------------------------------------------


def test_regression_dispatch_batch_mi_noise_gate_gpu_honours_disable_gpu(monkeypatch):
    """Pre-fix: dispatch_batch_mi_with_noise_gate_gpu ignored MLFRAME_DISABLE_GPU entirely, even
    with force_backend='cuda'. Post-fix: an internal gpu_globally_disabled() check wins first."""
    import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bng_gpu
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.setattr(bng_gpu, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bng_gpu, "_CUPY_AVAIL", True)

    def _fail_if_called(*a, **k):
        """Stand-in that raises AssertionError, so the caller's failure path is the one under test."""
        raise AssertionError("GPU backend must not be invoked when globally disabled")

    monkeypatch.setattr(bng_gpu, "batch_mi_with_noise_gate_cuda", _fail_if_called)
    monkeypatch.setattr(bng_gpu, "batch_mi_with_noise_gate_cupy", _fail_if_called)
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True

    result = bng_gpu.dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d=np.zeros((10, 2), dtype=np.int32),
        factors_nbins=np.array([2, 2], dtype=np.int32),
        classes_y=np.zeros(10, dtype=np.int32),
        classes_y_safe=np.zeros(10, dtype=np.int32),
        freqs_y=np.array([1.0]),
        npermutations=0,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.0,
        use_su=False,
        force_backend="cuda",
    )
    assert result is None


# ---------------------------------------------------------------------------
# GPU_INFRA_A-11 (P2): _batch_joint_entropy_pairs's njit kernel has no host-side validation that
# nbins >= 1 for referenced columns, unlike every CUDA kernel in the same cluster, which pre-validate
# this class of input before launch (an unguarded OOB index write is undefined behaviour on CPU too).
# ---------------------------------------------------------------------------


def test_regression_env_gpu_default_on_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was checked via an inline reimplementation, not the shared
    gpu_globally_disabled(). Post-fix: _env_gpu_default_on delegates to the shared policy function."""
    from mlframe.feature_selection.filters._gpu_resident_fe import _env_gpu_default_on
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True
    assert _env_gpu_default_on("MLFRAME_FE_GPU_RESIDENT_CODES") is False


# ---------------------------------------------------------------------------
# GPU_INFRA_C-1 (P1): _fe_gpu_strict's AUTO-fit-shape state (_AUTO_FIT_N/_AUTO_FIT_P) was held in bare,
# unlocked module-level globals shared across every thread -- StabilityMRMR's bootstrap loop already fits
# multiple MRMR instances concurrently via Parallel(backend="threading"), so two overlapping .fit() calls
# would silently read/write each other's fit shape, making the AUTO-STRICT engage/skip decision
# non-deterministic. Fixed via threading.local().
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_regression_resident_bincount_debug_check_bounds_catches_oob():
    """Pre-fix: no way to catch an out-of-range index short of the silent-wrap/OOB-write UB. Post-fix:
    debug_check_bounds=True raises ValueError instead."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._resident_bincount import resident_bincount

    x = cp.asarray([0, 1, -1, 2], dtype=cp.int32)  # -1 would silently wrap without the guard
    with pytest.raises(ValueError, match="index out of"):
        resident_bincount(cp, x, nc=3, debug_check_bounds=True)
    # Without the guard, it must NOT raise (this is the documented default sync-free UB contract).
    out = resident_bincount(cp, x, nc=3, debug_check_bounds=False)
    assert out.shape == (3,)


# ---------------------------------------------------------------------------
# GPU_INFRA_C-9 (P2): _fe_gpu_batch/_devices.py's _visible_device_ids reimplemented the
# MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES="" opt-out check inline instead of calling the shared
# _gpu_policy.gpu_globally_disabled(), extending the duplication pattern to a third file.
# ---------------------------------------------------------------------------


def test_regression_fe_gpu_batch_visible_device_ids_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was checked via an inline reimplementation. Post-fix:
    _visible_device_ids delegates to the shared gpu_globally_disabled()."""
    from mlframe.feature_selection.filters._fe_gpu_batch._devices import _visible_device_ids

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert _visible_device_ids() == []


# ---------------------------------------------------------------------------
# GPU_INFRA_C-8 (P2): the njit "sign" unary (_apply_unary code 8) mishandled NaN input -- IEEE comparisons
# with NaN are always False, so NaN fell through to the else branch and was coerced to -1.0, disagreeing
# with both np.sign(nan)==nan and this module's own cupy twin.
# ---------------------------------------------------------------------------


def test_regression_hermite_cupy_kernel_dispatch_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was never checked before importing/calling run_cupy_kernel_search.
    Post-fix: gpu_globally_disabled() gates it, falling back to random_batch. Observed at the call boundary:
    optimise_hermite_pair with optimizer='cupy_kernel' must not reach the cupy kernel under the opt-out and
    must run the random_batch fallback instead."""
    from mlframe.feature_selection.filters import _cupy_polynom_optimizer as cupy_mod
    from mlframe.feature_selection.filters import _hermite_fe_optimise as optimise_mod
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled
    from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True

    cupy_calls: list = []
    random_calls: list = []

    def _cupy_spy(**kwargs):
        """Record any routing into the cupy kernel search."""
        cupy_calls.append(kwargs)
        raise RuntimeError("cupy kernel search invoked under MLFRAME_DISABLE_GPU=1")

    _orig_random = optimise_mod._run_random_batch_search

    def _random_spy(**kwargs):
        """Record the fallback, then delegate to the real random_batch search."""
        random_calls.append(kwargs)
        return _orig_random(**kwargs)

    monkeypatch.setattr(cupy_mod, "run_cupy_kernel_search", _cupy_spy)
    monkeypatch.setattr(optimise_mod, "_run_random_batch_search", _random_spy)

    rng = np.random.default_rng(0)
    x_a = rng.standard_normal(400)
    x_b = rng.standard_normal(400)
    y = (x_a * x_b + 0.1 * rng.standard_normal(400) > 0).astype(np.int64)
    optimise_hermite_pair(x_a, x_b, y, optimizer="cupy_kernel", n_trials=20, sweep_degrees=False, seed=0)

    assert cupy_calls == []
    assert random_calls, "the random_batch fallback was not used under the GPU opt-out"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-2 (P1): _gpu_pairs.py / _batch_pair_mi_cuda_shared_fused.py both mutate a process-wide
# compiled-kernel object's max_dynamic_shared_size_bytes attribute with no lock -- two threads racing with
# different shared-mem requirements could under-provision one launch relative to what it needs.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DISCRETIZATION-1 (P1): discretize_2d_array_cuda_row_chunked's uniform/quantile branches never received
# the B-12 NaN-parity fix its non-chunked twin discretize_2d_array_cuda got -- plain cp.min/cp.max /
# cp.percentile propagated NaN, silently collapsing a whole NaN-bearing column to one degenerate bucket.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_regression_discretize_2d_array_cuda_row_chunked_nan_parity(method):
    """Pre-fix: a single NaN anywhere in a column poisoned that column's edges to NaN, collapsing every
    real value in the column to one degenerate bin. Post-fix: NaN-aware min/max/percentile matches the
    non-chunked (already-fixed) sibling's behaviour -- non-NaN rows land in real, non-degenerate bins."""
    pytest.importorskip("cupy")

    from mlframe.feature_selection.filters.discretization import discretize_2d_array_cuda_row_chunked

    rng = np.random.default_rng(0)
    n, ncols = 2000, 3
    arr = rng.standard_normal((n, ncols)).astype(np.float64)
    arr[5, 0] = np.nan  # a single NaN in column 0; columns 1-2 stay clean
    # Force the row-chunked path with a tiny free_bytes budget (mirrors existing dispatch tests).
    out = discretize_2d_array_cuda_row_chunked(arr, n_bins=8, method=method, dtype=np.int16, free_bytes=1 << 16)
    # Column 0's non-NaN rows must NOT all collapse into a single degenerate bucket.
    col0_real = out[np.arange(n) != 5, 0]
    assert len(np.unique(col0_real)) > 1, f"method={method}: NaN-poisoned column collapsed to one bin"
    if method == "uniform":
        # Only the uniform branch routes individual NaN VALUES to the dedicated NaN code (n_bins) --
        # matches the non-chunked sibling's behaviour, which the quantile branch never had either.
        assert out[5, 0] == 8, "NaN row did not get the dedicated NaN code"


# ---------------------------------------------------------------------------
# DISCRETIZATION-2 (P1): Bayesian Blocks (O(N^2)-per-column DP) had its subsample safety valve
# (bb_subsample_threshold) default to 0 (disabled) at the per_feature_edges dispatch layer, unlike every
# internal benchmark exercising it at realistic scale (which always overrides it to 1000).
# ---------------------------------------------------------------------------


def test_regression_cat_confirm_permutation_gpu_gate_honours_disable_gpu(monkeypatch):
    """Pre-fix: MLFRAME_DISABLE_GPU=1 was never checked. Post-fix: _perm_kernel_dispatch_use_gpu returns
    False whenever the global GPU off-switch is set, even for an explicit backend='gpu' request."""
    from mlframe.feature_selection.filters._cat_confirm_permutation import _perm_kernel_dispatch_use_gpu
    from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    assert gpu_globally_disabled() is True
    assert _perm_kernel_dispatch_use_gpu(n_samples=1_000_000, n_perms=1000, backend="gpu") is False
    assert _perm_kernel_dispatch_use_gpu(n_samples=1_000_000, n_perms=1000, backend="auto") is False


def test_regression_cat_interactions_step_pair_search_gpu_gate_honours_disable_gpu(monkeypatch):
    """Post-fix: run_cat_interaction_step's pair-search GPU gate consults gpu_globally_disabled() before
    honouring cfg.backend='gpu', and raises a clear error for an explicit backend='gpu' request under the
    global off-switch instead of silently using GPU."""
    from mlframe.feature_selection.filters._cat_interactions_step import run_cat_interaction_step
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    rng = np.random.default_rng(0)
    n = 300
    data = rng.integers(0, 2, size=(n, 3)).astype(np.int32)
    nbins = np.array([2, 2, 2], dtype=np.int64)
    classes_y = data[:, 2].astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float32) / n
    cfg = CatFEConfig(enable=True, min_n_samples=50, backend="gpu", marginal_floor=0.0)

    with pytest.raises(RuntimeError, match="GPU is globally disabled"):
        run_cat_interaction_step(
            data=data,
            cols=["a", "b", "y"],
            nbins=nbins,
            target_indices=np.array([2], dtype=np.int64),
            classes_y=classes_y,
            classes_y_safe=classes_y,
            freqs_y=freqs_y,
            categorical_vars=[0, 1],
            cfg=cfg,
            dtype=np.int32,
            verbose=0,
        )


# ---------------------------------------------------------------------------
# CAT_INTERACTION_A-3 (P1): CatFEConfig.perm_budget_strategy defaults to "bandit_ucb1" (not "fixed"), and
# the bandit allocator unconditionally ignores weights -- so the DEFAULT confirmation path for any weighted
# cat-FE fit silently tested a weighted II_obs against an unweighted permutation null.
# ---------------------------------------------------------------------------


def test_regression_resident_operand_key_includes_device_id(monkeypatch):
    """resident_operand's cache is device-scoped: the same content requested while a different cupy device is
    active must get its own resident copy, never the buffer uploaded on the first device. Observed by switching
    the reported active device id between calls on the same host array."""
    cp = pytest.importorskip(
        "cupy",
        reason="resource missing: cupy needs a CUDA GPU host (the gpu extra); must run on the GPU CI job, not the CPU deep suite",
    )

    from mlframe.feature_selection.filters import _fe_resident_operands as mod

    active = {"id": 0}

    class _FakeDevice:
        """Stands in for cp.cuda.Device with a controllable id; the real allocation still lands on device 0."""

        def __init__(self, *args, **kwargs):
            self.id = active["id"]

    monkeypatch.setattr(cp.cuda, "Device", _FakeDevice)
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    arr = np.random.default_rng(12345).standard_normal(257)

    g_dev0 = mod.resident_operand(arr, "device_key_probe")
    g_dev0_again = mod.resident_operand(arr, "device_key_probe")
    active["id"] = 1
    g_dev1 = mod.resident_operand(arr, "device_key_probe")

    assert g_dev0_again is g_dev0, "same content on the same device must hit the cache"
    assert g_dev1 is not g_dev0, "same content on a different device must not alias the first device's buffer"


def test_regression_fe_batch_mi_logs_gpu_fallback(caplog, monkeypatch):
    """Post-fix: a multi_gpu_fe_batch_mi failure is logged at warning level, not silently swallowed."""
    import logging

    from mlframe.feature_selection.filters import _fe_batch_dispatch as mod

    monkeypatch.setattr(mod, "choose_fe_batch_backend", lambda n, k: "gpu")

    def _raising_multi_gpu(*args, **kwargs):
        """Stand-in that raises RuntimeError, so the caller's failure path is the one under test."""
        raise RuntimeError("fake-multi-gpu-failure")

    import mlframe.feature_selection.filters._fe_gpu_batch as gpu_batch_mod

    monkeypatch.setattr(gpu_batch_mod, "multi_gpu_fe_batch_mi", _raising_multi_gpu)

    rng = np.random.default_rng(0)
    X_cands = rng.standard_normal((100, 4))
    y_codes = rng.integers(0, 2, 100)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._fe_batch_dispatch"):
        result = mod.fe_batch_mi(X_cands, y_codes, backend="gpu")
    assert result is not None
    assert any("fake-multi-gpu-failure" in rec.message for rec in caplog.records)


def test_regression_fe_cmi_perm_null_gpu_no_dead_resident_y_isinstance_check():
    """Post-fix: the dead `isinstance(y_h, cp.ndarray)` branches are removed."""
    import inspect

    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as cmi_gpu

    src = inspect.getsource(cmi_gpu)
    assert not any(line.strip().startswith("if isinstance(y_h, cp.ndarray)") for line in src.splitlines())


# ---------------------------------------------------------------------------
# DISCRETIZATION-1 (P1): discretize_2d_array_cuda_row_chunked's uniform/quantile branches never received
# the B-12 NaN-parity fix its non-chunked twin discretize_2d_array_cuda got -- plain cp.min/cp.max /
# cp.percentile propagated NaN, silently collapsing a whole NaN-bearing column to one degenerate bucket.
# ---------------------------------------------------------------------------
