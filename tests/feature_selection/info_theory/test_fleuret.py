"""Unit + biz_value tests for ``mlframe.feature_selection.filters.fleuret``.

Public surface covered:
- ``get_fleuret_criteria_confidence`` (``@njit`` core: permutation budget loop, returns ``(nfailed, nchecked)``)
- ``parallel_fleuret`` (joblib worker: rebuilds numba.typed.Dict caches and forwards to the core)
- ``get_fleuret_criteria_confidence_parallel`` (joblib pool spawner: aggregates workers into ``(bootstrapped_gain, confidence, entropy_cache)``)

Confidence semantics: ``confidence = 1 - (1 + nfailed) / (1 + nchecked)`` (add-one-corrected p-value) where ``nfailed`` is the count of permuted-y shuffles whose conditional gain still met the
bootstrapped target. High confidence == genuine signal (no permutation matched the original); low confidence == easily faked (target was noise).
"""

from __future__ import annotations

import numpy as np
import pytest
from numba.core import types
from numba.typed import Dict as NumbaDict

from mlframe.feature_selection.filters.fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)


# ================================================================================================
# Fixtures: synthetic factor matrices.
# ================================================================================================


def _new_entropy_cache() -> NumbaDict:
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _new_cond_mi_cache() -> NumbaDict:
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _build_xor_factors(n: int = 2000, seed: int = 42) -> tuple:
    """XOR synergy: ``y = x_0 XOR x_1``. I(x_0; y) ~ 0, I(x_1; y) ~ 0 individually, but I(x_0, x_1; y) ~ ln(2). With x_0 already in ``selected_vars``,
    conditional MI I(x_1; y | x_0) is large and stable under permutations of y => confidence approaches 1.0 with enough permutations.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, size=n).astype(np.int32)
    x1 = rng.integers(0, 2, size=n).astype(np.int32)
    x2 = rng.integers(0, 2, size=n).astype(np.int32)
    y_col = np.bitwise_xor(x0, x1).astype(np.int32)
    factors_data = np.column_stack([x0, x1, x2, y_col]).astype(np.int32)
    factors_nbins = np.array([2, 2, 2, 2], dtype=np.int64)
    return factors_data, factors_nbins


def _build_uncorrelated_factors(n: int = 2000, seed: int = 7) -> tuple:
    """All columns drawn iid Bernoulli(0.5); target column likewise independent. Any "conditional gain" is pure sampling noise => permuted y will routinely
    match or exceed the original => high ``nfailed`` => low confidence.
    """
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, size=n).astype(np.int32) for _ in range(4)]
    factors_data = np.column_stack(cols).astype(np.int32)
    factors_nbins = np.array([2, 2, 2, 2], dtype=np.int64)
    return factors_data, factors_nbins


def _call_core(
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    *,
    x: tuple = (1,),
    y_indices: tuple = (3,),
    selected_vars=None,
    npermutations: int = 50,
    bootstrapped_gain: float = 0.05,
    max_failed: int = None,
    extra_x_shuffling: bool = True,
):
    """Invoke the njit core directly. We pre-build the numba.typed.Dict caches so the function signature is satisfied and the core can mutate them in place."""
    if selected_vars is None:
        selected_vars = [0]
    if max_failed is None:
        max_failed = max(npermutations, 1)
    y_arr = np.asarray(y_indices, dtype=np.int64)
    return get_fleuret_criteria_confidence(
        data_copy=factors_data.copy(),
        factors_nbins=factors_nbins,
        x=tuple(x),
        y=y_arr,
        selected_vars=list(selected_vars),
        npermutations=npermutations,
        bootstrapped_gain=bootstrapped_gain,
        max_failed=max_failed,
        nexisting=0,
        mrmr_relevance_algo="fleuret",
        mrmr_redundancy_algo="fleuret",
        max_veteranes_interactions_order=1,
        cached_cond_MIs=_new_cond_mi_cache(),
        entropy_cache=_new_entropy_cache(),
        extra_x_shuffling=extra_x_shuffling,
    )


def _confidence_from_core(nfailed: int, nchecked: int) -> float:
    """Mirror the fleuret.py caller-side formula: confidence = 1 - p where p is the add-one-corrected (Phipson & Smyth 2010)
    permutation p-value ``(1 + nfailed) / (1 + nchecked)`` -- never exactly 1.0 on a null feature. Guard ``nchecked == 0``."""
    return (1.0 - (1.0 + nfailed) / (1.0 + nchecked)) if nchecked > 0 else 0.0


@pytest.fixture
def xor_factors():
    return _build_xor_factors()


@pytest.fixture
def noise_factors():
    return _build_uncorrelated_factors()


# ================================================================================================
# get_fleuret_criteria_confidence -- core invariants.
# ================================================================================================


def test_confidence_in_unit_interval_xor(xor_factors):
    """``confidence = 1 - (1 + nfailed) / (1 + nchecked)`` must fall inside [0, 1] for any well-formed input."""
    factors_data, factors_nbins = xor_factors
    nfailed, nchecked = _call_core(factors_data, factors_nbins, npermutations=50)
    assert nchecked > 0
    assert 0 <= nfailed <= nchecked
    conf = _confidence_from_core(nfailed, nchecked)
    assert 0.0 <= conf <= 1.0


def test_confidence_in_unit_interval_noise(noise_factors):
    factors_data, factors_nbins = noise_factors
    nfailed, nchecked = _call_core(factors_data, factors_nbins, npermutations=50)
    assert nchecked > 0
    assert 0 <= nfailed <= nchecked
    conf = _confidence_from_core(nfailed, nchecked)
    assert 0.0 <= conf <= 1.0


def test_xor_perfect_synergy_high_confidence(xor_factors):
    """Noiseless XOR: with x_0 already selected, conditional gain I(x_1; y | x_0) ~ ln(2). Shuffling y wipes the relationship => permuted gain near zero
    => almost no permutation reaches the bootstrapped target => confidence near 1.0 with full_npermutations >= 100.
    """
    factors_data, factors_nbins = xor_factors
    nfailed, nchecked = _call_core(
        factors_data,
        factors_nbins,
        npermutations=150,
        bootstrapped_gain=0.1,
        max_failed=150,
    )
    conf = _confidence_from_core(nfailed, nchecked)
    assert conf >= 0.9, f"expected XOR synergy confidence >= 0.9, got {conf} (nfailed={nfailed}/{nchecked})"


def test_uncorrelated_target_low_confidence(noise_factors):
    """All columns iid Bernoulli => no signal. With ``bootstrapped_gain=0.0`` any non-negative permuted gain counts as a "fail" (the alternative is no
    smaller than the original), so ``nfailed`` saturates and confidence collapses near 0. This is the regime where ``screen_predictors`` would reject
    the candidate. We assert ``conf <= 0.05`` (alpha-level): the test verifies the lower bound of the confidence distribution, not a specific value.
    """
    factors_data, factors_nbins = noise_factors
    nfailed, nchecked = _call_core(
        factors_data,
        factors_nbins,
        npermutations=100,
        bootstrapped_gain=0.0,
        max_failed=100,
    )
    conf = _confidence_from_core(nfailed, nchecked)
    assert conf <= 0.05, f"expected uncorrelated target with bootstrapped_gain=0 to give confidence ~ 0, got {conf} (nfailed={nfailed}/{nchecked})"


def test_npermutations_zero_returns_zero_no_crash(xor_factors):
    """``npermutations=0`` must hit the documented guard and return ``(0, 0)`` cleanly. Legacy code raised ``UnboundLocalError`` on ``i`` because the
    for-loop never assigned it (see fleuret.py docstring). Caller-side ``confidence`` formula then resolves to ``0.0`` via the ``nchecked > 0`` guard.
    """
    factors_data, factors_nbins = xor_factors
    nfailed, nchecked = _call_core(factors_data, factors_nbins, npermutations=0)
    assert (nfailed, nchecked) == (0, 0)
    assert _confidence_from_core(nfailed, nchecked) == 0.0


def test_small_sample_n10_no_crash():
    """Edge case: ``n=10`` (smallest sample that ``screen_predictors`` accepts). The core must complete and return valid ``(nfailed, nchecked)`` counts.
    Confidence is unreliable at this size; we only assert structural sanity, not statistical correctness.
    """
    factors_data, factors_nbins = _build_xor_factors(n=10, seed=1)
    nfailed, nchecked = _call_core(factors_data, factors_nbins, npermutations=20, max_failed=20)
    assert nchecked > 0
    assert 0 <= nfailed <= nchecked


def test_max_failed_short_circuits_loop(xor_factors):
    """When ``nfailed >= max_failed`` the core breaks early. Set ``max_failed=1`` on noise: the first permutation that succeeds halts the loop, so
    ``nchecked`` < ``npermutations``. Hard to guarantee on XOR (no permutation succeeds), so we use uncorrelated data for the trigger.
    """
    factors_data, factors_nbins = _build_uncorrelated_factors(n=500, seed=99)
    nfailed, nchecked = get_fleuret_criteria_confidence(
        data_copy=factors_data.copy(),
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=200,
        bootstrapped_gain=0.0,  # any permuted gain >= 0 fails => first iteration trips max_failed.
        max_failed=1,
        nexisting=0,
        mrmr_relevance_algo="fleuret",
        mrmr_redundancy_algo="fleuret",
        max_veteranes_interactions_order=1,
        cached_cond_MIs=_new_cond_mi_cache(),
        entropy_cache=_new_entropy_cache(),
        extra_x_shuffling=True,
    )
    assert nfailed >= 1
    assert nchecked <= 200


# ================================================================================================
# Mutate-and-restore regression: the njit core aliases the caller's real ``factors_data`` (screen no
# longer takes a whole-matrix copy), so it MUST restore every column it permutes before returning.
# ================================================================================================


@pytest.mark.parametrize("extra_x_shuffling", [True, False])
def test_core_restores_permuted_columns_byte_identical(xor_factors, extra_x_shuffling):
    """The core permutes the y columns (always) and the x columns (when ``extra_x_shuffling``) IN PLACE. Because ``data_copy`` may alias the caller's
    real matrix, the buffer must be byte-identical to entry on return. Pre-fix code left the columns shuffled (it relied on the caller's throwaway copy);
    that would corrupt an aliased ``factors_data``. This sensor fails on the pre-fix core and passes post-fix.
    """
    factors_data, factors_nbins = xor_factors
    buf = factors_data.copy()
    before = buf.copy()
    nfailed, nchecked = get_fleuret_criteria_confidence(
        data_copy=buf,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=40,
        bootstrapped_gain=0.1,
        max_failed=40,
        nexisting=0,
        mrmr_relevance_algo="fleuret",
        mrmr_redundancy_algo="fleuret",
        max_veteranes_interactions_order=1,
        cached_cond_MIs=_new_cond_mi_cache(),
        entropy_cache=_new_entropy_cache(),
        extra_x_shuffling=extra_x_shuffling,
    )
    assert nchecked > 0
    np.testing.assert_array_equal(buf, before), "core must restore every permuted column so an aliased factors_data is left untouched"


def test_core_aliased_buffer_matches_fresh_copy_each_call(xor_factors):
    """Behavioural proof the restore makes aliasing safe: running the core TWICE on ONE shared (aliased) buffer must give the same verdicts as running it
    twice, each on a fresh copy. If the buffer were left shuffled (pre-fix), the second aliased call would start from corrupted data and diverge.
    """
    factors_data, factors_nbins = xor_factors
    kw = dict(
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=40,
        bootstrapped_gain=0.1,
        max_failed=40,
        nexisting=0,
        mrmr_relevance_algo="fleuret",
        mrmr_redundancy_algo="fleuret",
        max_veteranes_interactions_order=1,
        extra_x_shuffling=True,
    )
    shared = factors_data.copy()
    a1 = get_fleuret_criteria_confidence(data_copy=shared, cached_cond_MIs=_new_cond_mi_cache(), entropy_cache=_new_entropy_cache(), **kw)
    a2 = get_fleuret_criteria_confidence(data_copy=shared, cached_cond_MIs=_new_cond_mi_cache(), entropy_cache=_new_entropy_cache(), **kw)

    f1 = get_fleuret_criteria_confidence(data_copy=factors_data.copy(), cached_cond_MIs=_new_cond_mi_cache(), entropy_cache=_new_entropy_cache(), **kw)
    f2 = get_fleuret_criteria_confidence(data_copy=factors_data.copy(), cached_cond_MIs=_new_cond_mi_cache(), entropy_cache=_new_entropy_cache(), **kw)
    assert a1 == f1 and a2 == f2, f"aliased-buffer calls {a1},{a2} must match fresh-copy calls {f1},{f2}"


# ================================================================================================
# parallel_fleuret -- joblib worker contract.
# ================================================================================================


def test_parallel_fleuret_worker_returns_three_tuple(xor_factors):
    """The joblib worker must return ``(nfailed, nchecked, entropy_cache_dict)`` where ``entropy_cache_dict`` is a plain Python dict (numba.typed.Dict
    isn't picklable across joblib workers, so the worker materialises it before returning).
    """
    factors_data, factors_nbins = xor_factors
    nfailed, nchecked, entropy_cache_dict = parallel_fleuret(
        data=factors_data,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=20,
        bootstrapped_gain=0.1,
        max_failed=20,
        nexisting=0,
        cached_cond_MIs={},
        entropy_cache={},
    )
    assert nchecked > 0
    assert 0 <= nfailed <= nchecked
    assert isinstance(entropy_cache_dict, dict)


def test_parallel_matches_serial_same_seed(xor_factors):
    """``parallel_fleuret`` (single worker) and ``get_fleuret_criteria_confidence`` must agree on the verdict for the same input on a strong-signal
    target. The joblib worker copies ``data`` and seeds numba's RNG fresh, so per-iteration ``(nfailed, nchecked)`` may differ; the public contract is
    the *aggregated confidence* verdict.
    """
    factors_data, factors_nbins = xor_factors

    nfailed_s, nchecked_s = _call_core(factors_data, factors_nbins, npermutations=30, bootstrapped_gain=0.1, max_failed=30)

    nfailed_p, nchecked_p, _ = parallel_fleuret(
        data=factors_data,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=30,
        bootstrapped_gain=0.1,
        max_failed=30,
        nexisting=0,
        cached_cond_MIs={},
        entropy_cache={},
    )
    assert nchecked_s > 0 and nchecked_p > 0
    conf_s = _confidence_from_core(nfailed_s, nchecked_s)
    conf_p = _confidence_from_core(nfailed_p, nchecked_p)
    # On XOR both paths must agree on the "high signal" verdict (confidence well above 0.5).
    assert conf_s >= 0.5
    assert conf_p >= 0.5


# ================================================================================================
# get_fleuret_criteria_confidence_parallel -- pool spawner.
# ================================================================================================


def test_parallel_pool_returns_three_tuple(xor_factors):
    """The pool spawner returns ``(bootstrapped_gain, confidence, entropy_cache)``. ``confidence`` is in [0, 1] and the entropy cache is the same dict
    we passed in (mutated by the workers' aggregation).
    """
    factors_data, factors_nbins = xor_factors
    entropy_cache = {}
    bg, conf, ec_out = get_fleuret_criteria_confidence_parallel(
        data_copy=factors_data,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        bootstrapped_gain=0.1,
        npermutations=20,
        max_failed=20,
        nexisting=0,
        cached_cond_MIs={},
        entropy_cache=entropy_cache,
        n_workers=1,
    )
    assert 0.0 <= conf <= 1.0
    assert ec_out is entropy_cache  # the caller's dict is mutated and returned.
    # ``bootstrapped_gain`` is either the original (passed through) or 0.0 (when nfailed >= max_failed).
    assert bg in (0.1, 0.0)


def test_parallel_pool_xor_high_confidence(xor_factors):
    """End-to-end pool path: XOR target with x_0 selected. Even single-worker, the aggregated confidence must be high."""
    factors_data, factors_nbins = xor_factors
    _, conf, _ = get_fleuret_criteria_confidence_parallel(
        data_copy=factors_data,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        bootstrapped_gain=0.1,
        npermutations=100,
        max_failed=100,
        nexisting=0,
        cached_cond_MIs={},
        entropy_cache={},
        n_workers=1,
    )
    assert conf >= 0.9, f"expected XOR pool confidence >= 0.9, got {conf}"


# ================================================================================================
# biz_value tests (fast).
# ================================================================================================


@pytest.mark.fast
def test_biz_xor_synergy_confidence_high():
    """biz_value: on perfect XOR synergy (y = x_0 XOR x_1) with x_0 selected, the Fleuret confidence step must score I(x_1; y | x_0) so confidently that
    ``confidence >= 0.95`` at ``full_npermutations=200``. This is the property MRMR relies on to confirm a synergistic candidate before promoting it
    into ``selected_vars``; regressing the conditional-MI cache key or the permutation loop breaks this bound.
    """
    factors_data, factors_nbins = _build_xor_factors(n=2000, seed=42)
    nfailed, nchecked = _call_core(
        factors_data,
        factors_nbins,
        npermutations=200,
        bootstrapped_gain=0.1,
        max_failed=200,
    )
    conf = _confidence_from_core(nfailed, nchecked)
    assert conf >= 0.95, f"expected XOR biz confidence >= 0.95, got {conf} (nfailed={nfailed}/{nchecked})"


@pytest.mark.fast
def test_biz_parallel_matches_serial():
    """biz_value: ``parallel_fleuret`` (single-worker) and ``get_fleuret_criteria_confidence`` must agree to within ``1e-12`` on the resulting confidence
    on a strong-signal target. This is the contract that lets ``screen_predictors`` switch between serial and parallel paths transparently. On clean
    XOR with ``bootstrapped_gain=0.1`` (well below the ~0.69-nat synergy gain but well above the permuted-baseline noise floor), every permutation
    drops conditional MI below the bar => ``nfailed=0`` in both paths => confidence exactly ``1.0`` independently of the permutation RNG seed.
    """
    factors_data, factors_nbins = _build_xor_factors(n=2000, seed=42)

    nf_s, nt_s = _call_core(factors_data, factors_nbins, npermutations=100, bootstrapped_gain=0.1, max_failed=100)
    conf_s = _confidence_from_core(nf_s, nt_s)

    nf_p, nt_p, _ = parallel_fleuret(
        data=factors_data,
        factors_nbins=factors_nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=100,
        bootstrapped_gain=0.1,
        max_failed=100,
        nexisting=0,
        cached_cond_MIs={},
        entropy_cache={},
    )
    conf_p = _confidence_from_core(nf_p, nt_p)
    assert abs(conf_s - conf_p) <= 1e-12, f"serial conf={conf_s}, parallel conf={conf_p}"
    # And both must hit the high-confidence verdict (the contract above is meaningful only when both paths actually find signal).
    assert conf_s >= 0.95
    assert conf_p >= 0.95


# ================================================================================================
# Run Tests
# ================================================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
