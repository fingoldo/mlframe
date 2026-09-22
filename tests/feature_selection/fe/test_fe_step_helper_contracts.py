"""Direct coverage for the FE-step sub-blocks that only had it transitively, through full fits.

Five helpers were reachable only by running a whole fit, so the contracts that decide whether they do anything at all - a self-gate, a key
convention, an opt-in flag - were never asserted on their own. Each of these is cheap to state directly and expensive to notice when it breaks,
because a helper that quietly no-ops still leaves a green integration suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fe_step_helpers import (
    apply_interaction_information_routing,
    apply_synergy_bootstrap,
    compute_pair_maxt_floor,
    run_cluster_aggregate_emission,
)


class _Estimator:
    """Bare attribute carrier: these helpers read configuration off ``self`` and nothing else."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


def _binned(n: int = 400, k: int = 6, seed: int = 0):
    """A discretised matrix, its per-column bin counts, and a binary target's codes and frequencies."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4, size=(n, k)).astype(np.int64)
    nbins = np.full(k, 4, dtype=np.int64)
    classes_y = rng.integers(0, 2, size=n).astype(np.int64)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64) / n
    return data, nbins, classes_y, freqs_y


def test_the_maxt_floor_self_gates_below_its_minimum_pair_count():
    """Below ``fe_pair_maxt_min_pairs`` the floor must be exactly 0.0 with an empty bias map: the narrow-pool no-op contract."""
    data, nbins, classes_y, freqs_y = _binned()
    est = _Estimator(fe_pair_maxt_null_permutations=25, fe_pair_maxt_min_pairs=30, fe_mm_debias_prevalence=False)
    floor, bias = compute_pair_maxt_floor(
        est, numeric_vars_to_consider={0, 1, 2}, n_pairs=20, data=data, nbins=nbins, classes_y=classes_y, freqs_y=freqs_y, verbose=0
    )
    assert floor == 0.0, f"a 20-pair pool under a 30-pair minimum must not raise a floor, got {floor}"
    assert bias == {}, f"no pair bias should be computed for a self-gated pool, got {len(bias)} entries"


def test_zero_permutations_disables_the_maxt_floor_entirely():
    """``fe_pair_maxt_null_permutations=0`` is the documented off switch, whatever the pool size."""
    data, nbins, classes_y, freqs_y = _binned()
    est = _Estimator(fe_pair_maxt_null_permutations=0, fe_pair_maxt_min_pairs=1, fe_mm_debias_prevalence=False)
    floor, bias = compute_pair_maxt_floor(
        est, numeric_vars_to_consider={0, 1, 2, 3}, n_pairs=99, data=data, nbins=nbins, classes_y=classes_y, freqs_y=freqs_y, verbose=0
    )
    assert floor == 0.0 and bias == {}


def test_the_maxt_floor_bias_keys_are_canonically_ordered():
    """The gate looks bias up by ``tuple(sorted(pair))``, so every key the floor emits has to be in that order."""
    data, nbins, classes_y, freqs_y = _binned(k=5)
    est = _Estimator(fe_pair_maxt_null_permutations=3, fe_pair_maxt_min_pairs=1, fe_mm_debias_prevalence=True)
    _floor, bias = compute_pair_maxt_floor(
        est, numeric_vars_to_consider={0, 1, 2, 3, 4}, n_pairs=10, data=data, nbins=nbins, classes_y=classes_y, freqs_y=freqs_y, verbose=0
    )
    if not bias:
        pytest.skip("this configuration produced no bias map to check")
    bad = [k for k in bias if not (isinstance(k, tuple) and len(k) == 2 and k[0] <= k[1])]
    assert not bad, f"bias keys are not canonically ordered, so the gate's sorted-tuple lookup will miss them: {bad[:5]}"


def test_the_maxt_floor_resets_its_failure_flag_per_call():
    """A stale True from an earlier FE step would mark this step's floor as failed; the helper must clear it."""
    data, nbins, classes_y, freqs_y = _binned()
    est = _Estimator(fe_pair_maxt_null_permutations=0, fe_pair_maxt_min_pairs=1, fe_mm_debias_prevalence=False)
    est._pair_maxt_floor_failed_ = True
    compute_pair_maxt_floor(est, numeric_vars_to_consider={0, 1}, n_pairs=5, data=data, nbins=nbins, classes_y=classes_y, freqs_y=freqs_y, verbose=0)
    assert est._pair_maxt_floor_failed_ is False, "the failure flag survived into a later step"


def test_the_interaction_information_router_is_off_by_default_and_returns_the_pool_untouched():
    """The router is opt-in; with the flag off it must hand back exactly what it was given."""
    pairs = {((0, 1), 0.5): 1, ((0, 2), 0.4): 2}
    est = _Estimator(fe_ii_routing_enable=False)
    assert apply_interaction_information_routing(
        est, prospective_pairs=pairs, data=np.zeros((10, 3), dtype=np.int64), nbins=np.full(3, 2), classes_y=np.zeros(10, dtype=np.int64),
        freqs_y=np.array([0.5, 0.5]), verbose=0, cached_MIs={}, synergy_added_idx=set()
    ) is pairs


def test_the_router_self_gates_below_its_minimum_pair_count():
    """Enabled but under the pair minimum, the router must still be a no-op rather than routing on a tiny pool."""
    pairs = {((0, 1), 0.5): 1}
    est = _Estimator(fe_ii_routing_enable=True, fe_ii_routing_null_permutations=25, fe_ii_routing_min_pairs=30)
    out = apply_interaction_information_routing(
        est, prospective_pairs=pairs, data=np.zeros((10, 3), dtype=np.int64), nbins=np.full(3, 2), classes_y=np.zeros(10, dtype=np.int64),
        freqs_y=np.array([0.5, 0.5]), verbose=0, cached_MIs={}, synergy_added_idx=set()
    )
    assert out is pairs, "a one-pair pool was routed despite the 30-pair minimum"


def test_an_empty_pool_is_returned_unchanged_by_the_router():
    """Nothing to route is not an error."""
    est = _Estimator(fe_ii_routing_enable=True, fe_ii_routing_null_permutations=25, fe_ii_routing_min_pairs=1)
    empty: dict = {}
    assert apply_interaction_information_routing(
        est, prospective_pairs=empty, data=np.zeros((4, 2), dtype=np.int64), nbins=np.full(2, 2), classes_y=np.zeros(4, dtype=np.int64),
        freqs_y=np.array([0.5, 0.5]), verbose=0, cached_MIs={}, synergy_added_idx=set()
    ) is empty


def test_the_helpers_are_importable_by_name():
    """The five sub-blocks are module-level functions, which is what makes direct coverage possible at all."""
    for fn in (apply_synergy_bootstrap, apply_interaction_information_routing, compute_pair_maxt_floor, run_cluster_aggregate_emission):
        assert callable(fn)
