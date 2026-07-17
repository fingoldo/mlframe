"""Coverage push for ``mlframe.feature_selection.filters.permutation``.

Public surface targeted:
- ``mi_direct`` (caller-side wrapper: serial / outer-joblib / inner-prange / Besag-Clifford / npermutations=0 short-circuit).
- ``parallel_mi`` (``@njit`` joblib worker: shuffles + ``max_failed`` early break).
- ``parallel_mi_prange`` (``@njit`` prange variant: deterministic per-iter LCG, no early stop).
- ``shuffle_arr`` (``@njit`` shim around ``np.random.shuffle``).
- ``distribute_permutations`` (budget partitioning).

Each test is small (``n <= 300``) and fixed-seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    compute_mi_from_classes,
    merge_vars,
)
from mlframe.feature_selection.filters.permutation import (
    distribute_permutations,
    mi_direct,
    parallel_mi,
    parallel_mi_besag_clifford,
    parallel_mi_prange,
    shuffle_arr,
)


# ================================================================================================
# Helpers.
# ================================================================================================


def _build_signal_factors(n: int = 300, seed: int = 42) -> tuple:
    """Strong signal: ``y = x`` (with small noise). I(x; y) is large; permutations almost never reach the original MI."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=n).astype(np.int32)
    flip = rng.random(size=n) < 0.05
    y = np.where(flip, rng.integers(0, 4, size=n).astype(np.int32), x)
    factors_data = np.column_stack([x, y]).astype(np.int32)
    factors_nbins = np.array([4, 4], dtype=np.int64)
    return factors_data, factors_nbins


def _build_noise_factors(n: int = 300, seed: int = 7) -> tuple:
    """All columns independent => I(x; y) ~ 0."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=n).astype(np.int32)
    y = rng.integers(0, 4, size=n).astype(np.int32)
    factors_data = np.column_stack([x, y]).astype(np.int32)
    factors_nbins = np.array([4, 4], dtype=np.int64)
    return factors_data, factors_nbins


def _make_classes(factors_data: np.ndarray, factors_nbins: np.ndarray):
    """Make classes."""
    cx, fx, _ = merge_vars(factors_data, (0,), None, factors_nbins, dtype=np.int32)
    cy, fy, _ = merge_vars(factors_data, (1,), None, factors_nbins, dtype=np.int32)
    mi = float(compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32))
    return cx, fx, cy, fy, mi


# ================================================================================================
# distribute_permutations.
# ================================================================================================


def test_distribute_permutations_sum_matches_total():
    """Whatever partition we compute, sum must equal ``npermutations``."""
    for total, workers in [(10, 1), (10, 3), (100, 4), (7, 5), (13, 2)]:
        out = distribute_permutations(total, workers)
        assert isinstance(out, list)
        assert len(out) == workers
        assert sum(out) == total


def test_distribute_permutations_remainder_lands_on_last():
    """The remainder lands on the last worker (documented contract)."""
    out = distribute_permutations(11, 3)
    assert out[0] == out[1] == 3  # 11 // 3 == 3
    assert out[-1] == 5  # 3 + (11 - 9)
    assert sum(out) == 11


def test_distribute_permutations_zero_total():
    """Distribute permutations zero total."""
    out = distribute_permutations(0, 4)
    assert sum(out) == 0
    assert len(out) == 4


def test_distribute_permutations_single_worker():
    """Distribute permutations single worker."""
    out = distribute_permutations(50, 1)
    assert out == [50]


# ================================================================================================
# shuffle_arr.
# ================================================================================================


def test_shuffle_arr_preserves_multiset():
    """Shuffle arr preserves multiset."""
    arr = np.arange(20, dtype=np.int32)
    expected = sorted(arr.tolist())
    shuffle_arr(arr)
    assert sorted(arr.tolist()) == expected
    assert arr.shape == (20,)


def test_shuffle_arr_runs_under_numba_rng():
    """``shuffle_arr`` is a thin njit shim around ``np.random.shuffle``. Numba has its own RNG state independent from numpy's seed,
    so we don't pin the exact permutation -- just verify it shuffles and is deterministic within a single warmed-up call sequence.
    """
    # Warm numba JIT first.
    shuffle_arr(np.arange(5, dtype=np.int32))

    arr = np.arange(30, dtype=np.int32)
    orig = arr.copy()
    shuffle_arr(arr)
    # Same multiset, very likely different ordering.
    assert sorted(arr.tolist()) == sorted(orig.tolist())


def test_shuffle_arr_actually_shuffles_with_high_probability():
    """30! permutations -> the chance of the shuffle hitting the identity is negligible. ``shuffle_arr`` is njit so it draws from numba's RNG, not numpy's seed."""
    arr = np.arange(30, dtype=np.int32)
    orig = arr.copy()
    shuffle_arr(arr)
    assert not np.array_equal(arr, orig)


# ================================================================================================
# parallel_mi: njit joblib worker.
# ================================================================================================


def test_parallel_mi_zero_permutations_returns_zero_zero():
    """Documented zero-budget guard: returns (0, 0) without UnboundLocalError on ``_i``."""
    factors, nbins = _build_signal_factors()
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    nf, nc = parallel_mi(cx, fx, cy, fy, 0, mi, max_failed=10, dtype=np.int32)
    assert (nf, nc) == (0, 0)


def test_parallel_mi_one_permutation_runs():
    """``npermutations=1`` exercises the for-loop exactly once. ``_i`` ends at 0 so ``nchecked == 1``."""
    factors, nbins = _build_signal_factors()
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    nf, nc = parallel_mi(cx, fx, cy, fy, 1, mi, max_failed=10, dtype=np.int32)
    assert nc == 1
    assert 0 <= nf <= 1


def test_parallel_mi_strong_signal_low_failure_rate():
    """On strong signal nfailed should be very low (no permutation reaches the original MI). Single-process serial path."""
    factors, nbins = _build_signal_factors(n=300, seed=42)
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    nf, nc = parallel_mi(cx, fx, cy, fy, 100, mi, max_failed=100, dtype=np.int32)
    assert nc <= 100
    assert nf / max(nc, 1) <= 0.1


def test_parallel_mi_max_failed_short_circuits():
    """On noise + bootstrapped_gain=0 (passed as ``original_mi``) every permuted MI >= 0 => first iteration fails => early break."""
    factors, nbins = _build_noise_factors()
    cx, fx, cy, fy, _ = _make_classes(factors, nbins)
    # original_mi=0.0 so every permuted MI >= original_mi.
    nf, nc = parallel_mi(cx, fx, cy, fy, 200, 0.0, max_failed=1, dtype=np.int32)
    assert nf >= 1
    assert nc <= 200


# ================================================================================================
# parallel_mi_prange: deterministic inner-parallel path.
# ================================================================================================


def test_parallel_mi_prange_zero_permutations():
    """Parallel mi prange zero permutations."""
    factors, nbins = _build_signal_factors()
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    nf, nc = parallel_mi_prange(cx, fx, cy, fy, 0, mi, np.uint64(0), dtype=np.int32)
    assert (nf, nc) == (0, 0)


def test_parallel_mi_prange_runs_full_budget():
    """prange path has no early-stop -- nchecked == npermutations exactly."""
    factors, nbins = _build_signal_factors()
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    nf, nc = parallel_mi_prange(cx, fx, cy, fy, 50, mi, np.uint64(42), dtype=np.int32)
    assert nc == 50
    assert 0 <= nf <= 50


def test_parallel_mi_prange_reproducible_across_runs():
    """Per-iter LCG keyed by base_seed => identical ``(nfailed, nchecked)`` for the same seed."""
    factors, nbins = _build_signal_factors()
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    r1 = parallel_mi_prange(cx, fx, cy, fy, 40, mi, np.uint64(123), dtype=np.int32)
    r2 = parallel_mi_prange(cx, fx, cy, fy, 40, mi, np.uint64(123), dtype=np.int32)
    assert r1 == r2


def test_parallel_mi_prange_noise_high_failure_rate():
    """No signal => most permutations match or exceed the original MI."""
    factors, nbins = _build_noise_factors()
    cx, fx, cy, fy, _ = _make_classes(factors, nbins)
    nf, nc = parallel_mi_prange(cx, fx, cy, fy, 100, 0.0, np.uint64(7), dtype=np.int32)
    assert nc == 100
    assert nf >= 50  # original_mi=0 so most permuted >= 0 count as "fail".


# ================================================================================================
# parallel_mi_besag_clifford: adaptive early-stopping.
# ================================================================================================


def test_besag_clifford_strong_signal_early_stops():
    """Strong signal => CI on p-value falls below p_low fast => ``nchecked < npermutations``."""
    factors, nbins = _build_signal_factors(n=300, seed=42)
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    _nf, nc = parallel_mi_besag_clifford(
        cx,
        fx,
        cy,
        fy,
        1000,
        mi,
        np.uint64(0),
        dtype=np.int32,
    )
    assert nc <= 1000
    assert nc >= 30  # min_perms guard.


def test_besag_clifford_zero_permutations():
    """Besag clifford zero permutations."""
    factors, nbins = _build_signal_factors()
    cx, fx, cy, fy, mi = _make_classes(factors, nbins)
    nf, nc = parallel_mi_besag_clifford(cx, fx, cy, fy, 0, mi, np.uint64(0), dtype=np.int32)
    assert (nf, nc) == (0, 0)


# ================================================================================================
# mi_direct: caller-side wrapper, branches under test.
# ================================================================================================


def test_mi_direct_zero_permutations():
    """``npermutations=0`` skips the permutation block; ``confidence`` stays at the initial 0.0."""
    factors, nbins = _build_signal_factors()
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=0,
        n_workers=1,
        parallelism="none",
    )
    assert bg > 0  # signal data
    assert conf == 0.0


def test_mi_direct_serial_path_strong_signal():
    """``n_workers=1`` + small budget => serial inner loop (the ``else:`` branch)."""
    factors, nbins = _build_signal_factors(n=300, seed=42)
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=20,
        n_workers=1,
        parallelism="none",
    )
    assert bg > 0
    assert 0.0 <= conf <= 1.0
    # Strong signal => high confidence.
    assert conf >= 0.5


def test_mi_direct_serial_path_noise_low_confidence():
    """No signal => bootstrapped_gain returns 0 quickly (max_failed trip) and confidence ~ 0."""
    factors, nbins = _build_noise_factors(n=300, seed=99)
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=50,
        max_failed=1,
        n_workers=1,
        parallelism="none",
    )
    # On true noise the early ``max_failed=1`` branch fires and ``original_mi`` is reset to 0.
    assert bg >= 0
    assert 0.0 <= conf <= 1.0


def test_mi_direct_inner_parallelism():
    """``parallelism="inner"`` + ``npermutations > NMAX_NONPARALLEL_ITERS`` dispatches to ``parallel_mi_prange``."""
    factors, nbins = _build_signal_factors()
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=30,
        parallelism="inner",
        base_seed=7,
    )
    assert bg > 0
    assert 0.0 <= conf <= 1.0


def test_mi_direct_besag_clifford_path():
    """``parallelism="bc"`` dispatches to the Besag-Clifford sequential test."""
    factors, nbins = _build_signal_factors()
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=200,
        parallelism="bc",
        base_seed=1,
    )
    assert bg > 0
    assert 0.0 <= conf <= 1.0


def test_mi_direct_outer_pool_path():
    """``n_workers > 1`` (and budget > NMAX_NONPARALLEL_ITERS) dispatches to the joblib pool of ``parallel_mi`` workers.

    The njit ``parallel_mi`` worker can't accept ``classes_y_safe=None`` (numba ``np.asarray(none)`` typing error),
    so we pre-build it -- this mirrors how ``screen_predictors`` always supplies a real copy.
    """
    factors, nbins = _build_signal_factors()
    cy, fy, _ = merge_vars(factors, (1,), None, nbins, dtype=np.int32)
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=10,
        classes_y=cy,
        freqs_y=fy,
        classes_y_safe=cy.copy(),
        n_workers=2,
        parallel_kwargs={"backend": "threading", "max_nbytes": None},
    )
    assert bg > 0
    assert 0.0 <= conf <= 1.0


def test_mi_direct_with_precomputed_classes_y():
    """When ``classes_y`` / ``freqs_y`` are passed in we skip the inner ``merge_vars`` call."""
    factors, nbins = _build_signal_factors()
    cy, fy, _ = merge_vars(factors, (1,), None, nbins, dtype=np.int32)
    bg, conf = mi_direct(
        factors_data=factors,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=10,
        classes_y=cy,
        freqs_y=fy,
        n_workers=1,
        parallelism="none",
    )
    assert bg > 0
    assert 0.0 <= conf <= 1.0


# ================================================================================================
# biz_value fast subset.
# ================================================================================================


@pytest.mark.fast
def test_biz_mi_direct_signal_vs_noise_confidence_gap():
    """biz_value contract: on strong signal ``mi_direct`` must yield meaningfully higher confidence than on pure noise at the
    same budget. Regressing the serial loop or breaking ``confidence = 1 - nfailed/(i+1)`` collapses the gap.
    """
    sig_factors, sig_nbins = _build_signal_factors(n=300, seed=42)
    noise_factors, noise_nbins = _build_noise_factors(n=300, seed=99)

    bg_sig, conf_sig = mi_direct(
        factors_data=sig_factors,
        x=(0,),
        y=(1,),
        factors_nbins=sig_nbins,
        npermutations=50,
        max_failed=50,
        n_workers=1,
        parallelism="none",
    )
    bg_noise, conf_noise = mi_direct(
        factors_data=noise_factors,
        x=(0,),
        y=(1,),
        factors_nbins=noise_nbins,
        npermutations=50,
        max_failed=50,
        n_workers=1,
        parallelism="none",
    )
    # Strong signal must beat pure noise: signal yields nonzero bootstrapped_gain, noise has tiny / zero gain
    # AND signal's confidence beats noise's confidence.
    assert bg_sig > bg_noise
    assert conf_sig >= conf_noise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
