"""Pin the njit Kalman-filter recurrence to bit-identity vs the Python reference.

`_kf_single_segment` was a pure-Python scalar `for t in range(T)` recurrence; it
is now an @njit kernel (`_kf_inner`, fastmath=False) — the documented DEFAULT
state-space FE path. This test guards that the njit output stays bit-identical
(exact ==, NaN-aware) to the reference Python loop across tricky shapes
(empty / single / all-NaN / no-NaN / with-prior-drift). A future fastmath flip
or op-reorder that perturbs a single ULP fails here.

Bench: feature_engineering/_benchmarks/bench_kalman_filter_njit.py (165-290x).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_engineering import bayesian as B


def _kf_reference(observations, prior_traj, *, transition_sigma, observation_sigma, initial_variance):
    """Verbatim pre-njit pure-Python baseline."""
    T = observations.size
    out = np.full((T, 5), np.nan, dtype=np.float64)
    if T == 0:
        return out
    Q = transition_sigma**2
    R = observation_sigma**2
    init_obs = float(observations[0]) if np.isfinite(observations[0]) else 0.0
    mean = init_obs
    var = float(initial_variance)
    for t in range(T):
        drift = 0.0
        if t > 0 and np.isfinite(prior_traj[t]) and np.isfinite(prior_traj[t - 1]):
            drift = float(prior_traj[t] - prior_traj[t - 1])
        mean_pred = mean + drift
        var_pred = var + Q
        if np.isfinite(observations[t]):
            innovation = float(observations[t] - mean_pred)
            innovation_var = var_pred + R
            K = var_pred / (innovation_var + 1e-12)
            mean = mean_pred + K * innovation
            var = (1.0 - K) * var_pred
            log_lik = -0.5 * (math.log(2.0 * math.pi * innovation_var) + (innovation * innovation) / innovation_var)
        else:
            mean = mean_pred
            var = var_pred
            innovation = math.nan
            innovation_var = math.nan
            log_lik = math.nan
        out[t, 0] = mean
        out[t, 1] = var
        out[t, 2] = innovation
        out[t, 3] = innovation_var
        out[t, 4] = log_lik
    return out


def _make(n, seed, nan_frac):
    """Helper: Make."""
    rng = np.random.default_rng(seed)
    obs = np.cumsum(rng.normal(0, 1, max(n, 1))).astype(np.float64)
    obs[rng.random(obs.size) < nan_frac] = np.nan
    prior = np.cumsum(rng.normal(0, 0.3, max(n, 1))).astype(np.float64)
    prior[rng.random(prior.size) < 0.02] = np.nan
    return obs[:n], prior[:n]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 37, 1000])
@pytest.mark.parametrize("nan_frac", [0.0, 0.05, 1.0])
def test_kf_single_segment_bit_identical_to_python(n, nan_frac):
    """Kf single segment bit identical to python."""
    if not B._NUMBA_AVAILABLE:
        pytest.skip("numba unavailable -> _kf_single_segment uses the Python path it is pinned against")
    obs, prior = _make(n, seed=n + int(nan_frac * 100), nan_frac=nan_frac)
    got = B._kf_single_segment(
        obs,
        prior,
        transition_sigma=0.5,
        observation_sigma=1.0,
        initial_variance=1.0,
    )
    ref = _kf_reference(
        obs,
        prior,
        transition_sigma=0.5,
        observation_sigma=1.0,
        initial_variance=1.0,
    )
    assert got.shape == ref.shape
    assert np.array_equal(got, ref, equal_nan=True), f"njit KF diverged from Python reference at n={n} nan_frac={nan_frac}"


def test_kalman_filter_posterior_1d_bit_identical_with_groups():
    """End-to-end public API: grouped KF output bit-identical to per-group reference."""
    if not B._NUMBA_AVAILABLE:
        pytest.skip("numba unavailable")
    rng = np.random.default_rng(7)
    n = 500
    obs = np.cumsum(rng.normal(0, 1, n)).astype(np.float64)
    obs[rng.random(n) < 0.05] = np.nan
    prior = np.cumsum(rng.normal(0, 0.3, n)).astype(np.float64)
    group_ids = rng.integers(0, 4, n)

    res = B.kalman_filter_posterior_1d(
        obs,
        prior,
        group_ids=group_ids,
        transition_sigma=0.5,
        observation_sigma=1.0,
        initial_variance=1.0,
    )

    # Rebuild expected via reference per group segment.
    from mlframe.feature_engineering.grouped import iter_group_segments

    sort_idx, starts, ends = iter_group_segments(group_ids)
    exp_mean = np.full(n, np.nan)
    exp_ll = np.full(n, np.nan)
    for s, e in zip(starts, ends):
        idx = sort_idx[s:e]
        ref = _kf_reference(
            obs[idx],
            prior[idx],
            transition_sigma=0.5,
            observation_sigma=1.0,
            initial_variance=1.0,
        )
        exp_mean[idx] = ref[:, 0]
        exp_ll[idx] = ref[:, 4]

    assert np.array_equal(res["mean"], exp_mean, equal_nan=True)
    assert np.array_equal(res["log_lik"], exp_ll, equal_nan=True)
