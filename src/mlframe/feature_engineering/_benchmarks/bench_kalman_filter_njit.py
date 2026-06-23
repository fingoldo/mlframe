"""Bench: njit the 1-D Kalman-filter scalar recurrence (`_kf_single_segment`).

Target: feature_engineering/bayesian.py::_kf_single_segment — the documented
DEFAULT state-space FE path (KF preferred over the bootstrap PF). It is a pure
Python `for t in range(T)` scalar recurrence (mean/var update + innovation +
Gaussian log_lik via math.log) writing into a (T,5) out array. Pure-Python
scalar loops over the whole series are interpreter-bound; an @njit kernel runs
the identical scalar arithmetic in machine code.

Identity gate: fastmath=False so the FMA/reassoc set is NOT enabled — the njit
body executes the exact same float64 ops in the same order, so output is
bit-identical (exact ==) to the Python baseline (math.log == np.log for the
same double). NaN predict-only branch preserved.

Run:
    python -m mlframe.feature_engineering._benchmarks.bench_kalman_filter_njit
"""
from __future__ import annotations

import math
import time

import numpy as np
import numba


# --------------------------------------------------------------------------
# OLD: verbatim current pure-Python baseline (HEAD:_kf_single_segment body).
# --------------------------------------------------------------------------
def _kf_old(observations, prior_traj, *, transition_sigma, observation_sigma,
            initial_variance):
    T = observations.size
    out = np.full((T, 5), np.nan, dtype=np.float64)
    if T == 0:
        return out
    Q = transition_sigma ** 2
    R = observation_sigma ** 2
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
            log_lik = -0.5 * (
                math.log(2.0 * math.pi * innovation_var)
                + (innovation * innovation) / innovation_var
            )
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


# --------------------------------------------------------------------------
# NEW: njit kernel, identical scalar arithmetic, fastmath=False (bit-identical).
# --------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=False)
def _kf_njit(observations, prior_traj, Q, R, initial_variance):
    T = observations.size
    out = np.full((T, 5), np.nan, dtype=np.float64)
    if T == 0:
        return out
    if np.isfinite(observations[0]):
        mean = observations[0]
    else:
        mean = 0.0
    var = initial_variance
    for t in range(T):
        drift = 0.0
        if t > 0 and np.isfinite(prior_traj[t]) and np.isfinite(prior_traj[t - 1]):
            drift = prior_traj[t] - prior_traj[t - 1]
        mean_pred = mean + drift
        var_pred = var + Q
        if np.isfinite(observations[t]):
            innovation = observations[t] - mean_pred
            innovation_var = var_pred + R
            K = var_pred / (innovation_var + 1e-12)
            mean = mean_pred + K * innovation
            var = (1.0 - K) * var_pred
            log_lik = -0.5 * (
                math.log(2.0 * math.pi * innovation_var)
                + (innovation * innovation) / innovation_var
            )
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


def _new_wrapper(obs, prior, *, transition_sigma, observation_sigma, initial_variance):
    return _kf_njit(obs, prior, transition_sigma ** 2, observation_sigma ** 2,
                    float(initial_variance))


def _make(n, seed=0, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    obs = np.cumsum(rng.normal(0, 1, n)).astype(np.float64)
    mask = rng.random(n) < nan_frac
    obs[mask] = np.nan
    prior = np.cumsum(rng.normal(0, 0.3, n)).astype(np.float64)
    pmask = rng.random(n) < 0.02
    prior[pmask] = np.nan
    return obs, prior


def _best_of(fn, *args, reps=7, **kw):
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    kw = dict(transition_sigma=0.5, observation_sigma=1.0, initial_variance=1.0)

    # Identity gate across shapes incl. tiny / all-nan / no-nan.
    print("=== identity gate (exact ==) ===")
    ok = True
    for n in (0, 1, 3, 100, 10_000):
        for nf in (0.0, 0.05, 1.0):
            obs, prior = _make(max(n, 1), seed=n + int(nf * 100), nan_frac=nf)
            obs, prior = obs[:n], prior[:n]
            a = _kf_old(obs, prior, **kw)
            b = _new_wrapper(obs, prior, **kw)
            same = np.array_equal(a, b, equal_nan=True)
            ok = ok and same
            if not same:
                print(f"  MISMATCH n={n} nan_frac={nf}")
    print("  all bit-identical:", ok)

    # warm njit
    o, p = _make(64)
    _new_wrapper(o, p, **kw)

    print("\n=== speed (best-of-7) ===")
    for n in (2_000, 10_000, 100_000):
        obs, prior = _make(n, seed=1)
        told = _best_of(_kf_old, obs, prior, **kw)
        tnew = _best_of(_new_wrapper, obs, prior, **kw)
        print(f"  n={n:>7}: OLD {told*1e3:8.3f} ms  NEW {tnew*1e3:8.3f} ms  "
              f"speedup {told/tnew:6.2f}x")


if __name__ == "__main__":
    main()
