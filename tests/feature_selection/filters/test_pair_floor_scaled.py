"""The order-2 maxT floor of a large pool from a row subsample, with the scaling error measured on a block of the pairs (``_pair_floor_scaled``)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _pair_floor_scaled as ps
from mlframe.feature_selection.filters import _permutation_null as pn

exact_floor = pn.pooled_pair_permutation_null_joint_mi_floor


def _pool(n=60000, cols=60, bins=8, ky=2, seed=0):
    """A pure-noise pool: integer codes, a target independent of them, every column pair."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, bins, (n, cols)).astype(np.int8)
    nb = np.full(cols, bins, dtype=np.int64)
    y = rng.integers(0, ky, n).astype(np.int64)
    ia, ib = (v.astype(np.int64) for v in np.triu_indices(cols, 1))
    return data, nb, ia, ib, y, np.bincount(y, minlength=ky) / n


@pytest.fixture
def small_pool_allowed(monkeypatch):
    """Let the scaling run on a pool small enough for a test: no row or pair floor, a smaller subsample and block, a block that may cost as much as it likes."""
    monkeypatch.setattr(ps, "MIN_ROWS", 20_000)
    monkeypatch.setattr(ps, "MIN_PAIRS", 100)
    monkeypatch.setattr(ps, "MIN_SUB_ROWS", 10_000)
    monkeypatch.setattr(ps, "MAX_SUB_SHARE", 0.5)
    monkeypatch.setattr(ps, "MAX_BLOCK_COST", 1e9)
    monkeypatch.setattr(ps, "BLOCK_MIN_PAIRS", 200)
    monkeypatch.setattr(ps, "BLOCK_SHARE", 0.2)


def _scaled(data, nb, ia, ib, y, fy, k=40, seed=0):
    """``scaled_pair_floor`` of the pool with the exact function and the occupied-cell counter."""
    return ps.scaled_pair_floor(exact_floor, data, nb, ia, ib, y, fy, n_permutations=k, quantile=0.95, random_seed=seed, k_joint_of_pairs=pn._pairwise_occupied_joint_k)


def test_a_small_pool_or_few_rows_is_left_to_the_exact_floor():
    """Below the row and pair floors the scaling does not run and says why."""
    data, nb, ia, ib, y, fy = _pool(n=30000, cols=40)
    got, info = _scaled(data, nb, ia, ib, y, fy)
    assert got is None and "small pool" in info["why"]


def test_the_switch_turns_the_scaling_off(monkeypatch, small_pool_allowed):
    """``MLFRAME_FE_PAIR_FLOOR_SCALED=0`` leaves every floor exact."""
    monkeypatch.setenv("MLFRAME_FE_PAIR_FLOOR_SCALED", "0")
    data, nb, ia, ib, y, fy = _pool()
    assert _scaled(data, nb, ia, ib, y, fy) == (None, {"why": "disabled"})


def test_the_scaled_floor_is_the_exact_one_within_the_noise_of_the_floor_itself(small_pool_allowed):
    """On pure-noise pools of 8 and 16 bins the scaled floor over the exact one (both from the same number of shuffles) averages 1 and stays inside the spread of the floor itself (a quantile of the shuffles' maxima)."""
    ratios = []
    for bins, seed in ((8, 0), (8, 1), (16, 0), (16, 1)):
        data, nb, ia, ib, y, fy = _pool(bins=bins, seed=seed)
        exact = exact_floor(data, nb, ia, ib, y, fy, n_permutations=60, quantile=0.95, random_seed=seed, mm_debias=True, scaled=False)
        got, info = _scaled(data, nb, ia, ib, y, fy, k=60, seed=seed)
        assert got is not None, info
        ratios.append(got / exact)
    assert 0.9 <= float(np.mean(ratios)) <= 1.1, ratios
    assert ratios and all(0.75 <= r <= 1.3 for r in ratios), ratios


@pytest.mark.parametrize("rho, expected_factor", [(1.0, 1.0), (1.05, 1.0), (1.4, 1.4), (0.7, 0.7)])
def test_a_scaling_error_outside_the_accepted_window_is_corrected_by_the_measured_ratio(monkeypatch, small_pool_allowed, rho, expected_factor):
    """Fake exact floors of a constant 1e-3: a block ratio of 1.0 or 1.05 is accepted as it is, 1.4 and 0.7 correct the floor by the ratio."""
    data, nb, ia, ib, y, fy = _pool()
    rows_seen = {}

    def fake_exact(sub_data, *args, **kwargs):
        """A floor that does not depend on the pool."""
        rows_seen["rows"] = int(sub_data.shape[0])
        return 1e-3

    monkeypatch.setattr(ps, "_ratio", lambda *a, **k: rho)
    got, info = ps.scaled_pair_floor(fake_exact, data, nb, ia, ib, y, fy, n_permutations=25, quantile=0.95, random_seed=0, k_joint_of_pairs=pn._pairwise_occupied_joint_k)
    assert got == pytest.approx(1e-3 * (rows_seen["rows"] / data.shape[0]) * expected_factor)
    assert info["factor"] == pytest.approx(expected_factor)


def test_a_scaling_error_too_large_to_correct_sends_the_caller_to_the_exact_floor(monkeypatch, small_pool_allowed):
    """A block ratio of 3.0 is past the correctable range: no scaled floor, and the reason says why."""
    data, nb, ia, ib, y, fy = _pool()
    monkeypatch.setattr(ps, "_ratio", lambda *a, **k: 3.0)
    got, info = ps.scaled_pair_floor(lambda *a, **k: 1e-3, data, nb, ia, ib, y, fy, n_permutations=25, quantile=0.95, random_seed=0, k_joint_of_pairs=pn._pairwise_occupied_joint_k)
    assert got is None and "too large to correct" in info["why"], info


def test_halves_of_the_block_that_disagree_send_the_caller_to_the_exact_floor(monkeypatch, small_pool_allowed):
    """Two ratios 25 percent apart are noise the correction cannot be trusted on: no scaled floor."""
    data, nb, ia, ib, y, fy = _pool()
    ratios = iter([1.2, 0.9])
    monkeypatch.setattr(ps, "_ratio", lambda *a, **k: next(ratios))
    got, info = ps.scaled_pair_floor(lambda *a, **k: 1e-3, data, nb, ia, ib, y, fy, n_permutations=25, quantile=0.95, random_seed=0, k_joint_of_pairs=pn._pairwise_occupied_joint_k)
    assert got is None and "disagree" in info["why"]


def test_a_subsample_that_is_not_much_smaller_than_the_data_is_not_used(monkeypatch, small_pool_allowed):
    """With 32 bins and 4 classes a pair table has 4096 cells and 12 rows per cell is 49152 rows: no scaling on 60000."""
    monkeypatch.setattr(ps, "MAX_SUB_SHARE", 0.2)
    data, nb, ia, ib, y, fy = _pool(bins=32, ky=4, cols=30)
    got, info = _scaled(data, nb, ia, ib, y, fy)
    assert got is None and "much smaller than the data" in info["why"]


def test_the_exact_function_takes_the_scaled_route_only_for_a_debiased_pool_and_scaled_not_false(monkeypatch, small_pool_allowed):
    """``mm_debias=False`` and ``scaled=False`` never reach the scaling; the default of a debiased pool does."""
    data, nb, ia, ib, y, fy = _pool()
    calls = []
    monkeypatch.setattr(ps, "scaled_pair_floor", lambda *a, **k: calls.append(1) or (None, {"why": "stub"}))
    kw = dict(n_permutations=10, quantile=0.95, random_seed=0)
    exact_floor(data, nb, ia, ib, y, fy, mm_debias=False, **kw)
    exact_floor(data, nb, ia, ib, y, fy, mm_debias=True, scaled=False, **kw)
    assert calls == []
    exact_floor(data, nb, ia, ib, y, fy, mm_debias=True, **kw)
    assert calls == [1]


def test_a_block_that_would_cost_too_much_against_the_pool_is_not_run(monkeypatch, small_pool_allowed):
    """The calibration block runs on the host kernel: when it would cost more than ``MAX_BLOCK_COST`` of the exact floor the exact floor is asked for."""
    monkeypatch.setattr(ps, "MAX_BLOCK_COST", 0.01)
    data, nb, ia, ib, y, fy = _pool()
    got, info = _scaled(data, nb, ia, ib, y, fy)
    assert got is None and "costs too much" in info["why"]
