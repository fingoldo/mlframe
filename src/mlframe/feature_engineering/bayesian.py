"""Bayesian state-space filtering features.

Currently ships:

* ``particle_filter_posterior`` -- bootstrap particle filter for a
  1-D state with Gaussian observation likelihood and Gaussian-random-
  walk transition model. Emits per-row posterior median + percentile
  bounds (p10 / p90) as features.

For Kalman filter on linear-Gaussian state-space see e.g.
``filterpy.kalman`` (mlframe doesn't ship its own Kalman -- the PF
covers the non-Gaussian / nonlinear case which is the harder problem).

Per-group support: pass ``group_ids`` so each group gets its own
particle ensemble (no state bleed across boundaries).
"""

from __future__ import annotations

__all__ = [
    "particle_filter_posterior",
    "kalman_filter_posterior_1d",
    "kalman_smoother_posterior_1d",
    "bocpd_features",
    "online_bayesian_linear_regression",
]

import math
from typing import Optional

import numpy as np
from scipy.special import gammaln as _gammaln

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    # Safe fastmath subset: reassoc/contract/arcp/afn enable FMA + reciprocal
    # approximation + reassociation, but EXCLUDE nnan/ninf so the explicit
    # NaN handling (predict-only branch on missing observations) stays
    # correct. Full fastmath=True sets nnan and silently breaks NaN checks.
    _FASTMATH = {"reassoc", "contract", "arcp", "afn"}

    @_numba.njit(cache=True, fastmath=_FASTMATH)
    def _bocpd_inner(seg, hazard, mu0, kappa0, alpha0, beta0,
                     out_p_change, out_expected_rl, out_max_rl, max_run_length):
        T = seg.size
        # Run-length posterior grows by 1 per step (O(T) state, O(T^2) total). ``max_run_length`` caps it: when
        # cur_len would exceed the cap, the (lowest-probability) tail of large run lengths is dropped and the
        # surviving head [0:cap] is renormalised. cap<=0 disables the cap (legacy uncapped behaviour). With a cap
        # at least as large as the stream's true max run length, output is unchanged.
        cap = T + 1
        if max_run_length > 0 and max_run_length + 1 < cap:
            cap = max_run_length + 1
        mu = np.empty(cap, dtype=np.float64); mu[0] = mu0
        kappa = np.empty(cap, dtype=np.float64); kappa[0] = kappa0
        alpha = np.empty(cap, dtype=np.float64); alpha[0] = alpha0
        beta = np.empty(cap, dtype=np.float64); beta[0] = beta0
        rl_probs = np.empty(cap, dtype=np.float64); rl_probs[0] = 1.0
        log_pred = np.empty(cap, dtype=np.float64)
        cur_len = 1
        for t in range(T):
            x = seg[t]
            if not np.isfinite(x):
                # Predict-only growth + change-point mass; suff stats unchanged.
                cp_mass = 0.0
                for r in range(cur_len):
                    cp_mass += rl_probs[r] * hazard
                # shift suff stats right (prepend prior); shift_top clamps the highest written index to cap-1 so a
                # full vector drops its (lowest-probability) largest-run-length tail instead of overflowing.
                shift_top = cur_len if cur_len < cap else cap - 1
                for r in range(shift_top, 0, -1):
                    mu[r] = mu[r - 1]; kappa[r] = kappa[r - 1]
                    alpha[r] = alpha[r - 1]; beta[r] = beta[r - 1]
                    rl_probs[r] = rl_probs[r - 1] * (1.0 - hazard)
                mu[0] = mu0; kappa[0] = kappa0
                alpha[0] = alpha0; beta[0] = beta0
                rl_probs[0] = cp_mass
                cur_len = shift_top + 1
                total = 0.0
                for r in range(cur_len):
                    total += rl_probs[r]
                inv_t = 1.0 / (total + 1e-300)
                for r in range(cur_len):
                    rl_probs[r] *= inv_t
            else:
                # Closed-form Student-t log-pdf per run length.
                log_pred_max = -1e300
                for r in range(cur_len):
                    df = 2.0 * alpha[r]
                    scale_sq = beta[r] * (kappa[r] + 1.0) / (alpha[r] * kappa[r])
                    half_df = 0.5 * df
                    z2 = (x - mu[r]) * (x - mu[r]) / scale_sq
                    lp = (math.lgamma(half_df + 0.5) - math.lgamma(half_df)
                          - 0.5 * math.log(math.pi * df * scale_sq)
                          - (half_df + 0.5) * math.log1p(z2 / df))
                    log_pred[r] = lp
                    if lp > log_pred_max:
                        log_pred_max = lp
                # Posterior over run length.
                cp = 0.0
                for r in range(cur_len):
                    cp += rl_probs[r] * math.exp(log_pred[r] - log_pred_max) * hazard
                # Update suff stats (right-shift, prepend prior); shift_top clamps the highest written index to cap-1
                # so a full vector drops its (lowest-probability) largest-run-length tail instead of overflowing.
                shift_top = cur_len if cur_len < cap else cap - 1
                for r in range(shift_top, 0, -1):
                    pr = r - 1
                    mu[r] = (kappa[pr] * mu[pr] + x) / (kappa[pr] + 1.0)
                    kappa[r] = kappa[pr] + 1.0
                    alpha[r] = alpha[pr] + 0.5
                    beta[r] = beta[pr] + 0.5 * kappa[pr] * (x - mu[pr]) * (x - mu[pr]) / (kappa[pr] + 1.0)
                    rl_probs[r] = rl_probs[pr] * math.exp(log_pred[pr] - log_pred_max) * (1.0 - hazard)
                mu[0] = mu0; kappa[0] = kappa0
                alpha[0] = alpha0; beta[0] = beta0
                rl_probs[0] = cp
                cur_len = shift_top + 1
                total = 0.0
                for r in range(cur_len):
                    total += rl_probs[r]
                inv_t = 1.0 / (total + 1e-300)
                for r in range(cur_len):
                    rl_probs[r] *= inv_t
            # Summaries.
            ex = 0.0
            mx_idx = 0
            mx_val = rl_probs[0]
            for r in range(cur_len):
                ex += r * rl_probs[r]
                if rl_probs[r] > mx_val:
                    mx_val = rl_probs[r]; mx_idx = r
            out_p_change[t] = rl_probs[0]
            out_expected_rl[t] = ex
            out_max_rl[t] = float(mx_idx)


    @_numba.njit(cache=True, fastmath=_FASTMATH)
    def _oblr_inner(y_arr, X_arr, prior_precision, noise_sigma,
                    out_pred_mean, out_pred_var, out_log_marg):
        n, k = X_arr.shape
        mu = np.zeros(k, dtype=np.float64)
        Sigma = np.eye(k, dtype=np.float64) / prior_precision
        noise_var = noise_sigma * noise_sigma
        Sx = np.empty(k, dtype=np.float64)
        for t in range(n):
            # Sx = Sigma @ x_t
            for i in range(k):
                s = 0.0
                for j in range(k):
                    s += Sigma[i, j] * X_arr[t, j]
                Sx[i] = s
            pred_mean = 0.0
            xSx = 0.0
            for i in range(k):
                pred_mean += X_arr[t, i] * mu[i]
                xSx += X_arr[t, i] * Sx[i]
            pred_var = xSx + noise_var
            out_pred_mean[t] = pred_mean
            out_pred_var[t] = pred_var
            y_t = y_arr[t]
            if np.isfinite(y_t):
                out_log_marg[t] = -0.5 * (
                    math.log(2.0 * math.pi * pred_var)
                    + (y_t - pred_mean) * (y_t - pred_mean) / pred_var
                )
                innovation = y_t - pred_mean
                inv_pv = 1.0 / pred_var
                for i in range(k):
                    Ki = Sx[i] * inv_pv
                    mu[i] += Ki * innovation
                    # Sigma -= outer(K, Sx)
                    for j in range(k):
                        Sigma[i, j] -= Ki * Sx[j]

    # --- group drivers ----------------------------------------------------
    # The per-segment recursion (_bocpd_inner / _oblr_inner) is inherently
    # sequential, but groups (e.g. wells / panels) are independent, so the
    # parallel axis is `prange` ACROSS groups. The serial and parallel
    # drivers share the same per-segment core; the dispatcher
    # (_dispatch_recursion_backend) routes between them by group count.

    @_numba.njit(cache=True)
    def _bocpd_groups_serial(obs_sorted, starts, ends, hazard,
                             mu0, kappa0, alpha0, beta0,
                             out_p, out_ex, out_max, max_run_length):
        for g in range(starts.size):
            s = starts[g]; e = ends[g]
            _bocpd_inner(obs_sorted[s:e], hazard, mu0, kappa0, alpha0, beta0,
                         out_p[s:e], out_ex[s:e], out_max[s:e], max_run_length)

    @_numba.njit(parallel=True, cache=True)
    def _bocpd_groups_parallel(obs_sorted, starts, ends, hazard,
                               mu0, kappa0, alpha0, beta0,
                               out_p, out_ex, out_max, max_run_length):
        for g in _numba.prange(starts.size):
            s = starts[g]; e = ends[g]
            _bocpd_inner(obs_sorted[s:e], hazard, mu0, kappa0, alpha0, beta0,
                         out_p[s:e], out_ex[s:e], out_max[s:e], max_run_length)

    @_numba.njit(cache=True)
    def _oblr_groups_serial(y_sorted, X_sorted, starts, ends,
                            prior_precision, noise_sigma,
                            out_pm, out_pv, out_lm):
        for g in range(starts.size):
            s = starts[g]; e = ends[g]
            _oblr_inner(y_sorted[s:e], X_sorted[s:e], prior_precision, noise_sigma,
                        out_pm[s:e], out_pv[s:e], out_lm[s:e])

    @_numba.njit(parallel=True, cache=True)
    def _oblr_groups_parallel(y_sorted, X_sorted, starts, ends,
                              prior_precision, noise_sigma,
                              out_pm, out_pv, out_lm):
        for g in _numba.prange(starts.size):
            s = starts[g]; e = ends[g]
            _oblr_inner(y_sorted[s:e], X_sorted[s:e], prior_precision, noise_sigma,
                        out_pm[s:e], out_pv[s:e], out_lm[s:e])

    # 1-D Kalman-filter scalar recurrence. fastmath=False keeps the exact op
    # order, so the (T,5) output is bit-identical to the Python reference (the
    # explicit isfinite NaN predict-only branch also needs nnan OFF).
    @_numba.njit(cache=True, fastmath=False)
    def _kf_inner(observations, prior_traj, Q, R, initial_variance):
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


def _pf_single_segment(
    observations: np.ndarray,
    prior: np.ndarray,
    *,
    n_particles: int,
    transition_sigma: float,
    observation_sigma: float,
    resample_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one PF pass over a 1-D observation sequence.

    Returns ``(n_obs, 5)`` array of (p10, p50, p90, post_mean, neff)
    summaries at each timestep.

    Algorithm: bootstrap PF (Gordon 1993) with systematic resampling.
    State transition is a Gaussian random walk on top of ``prior``
    drift: ``x_t ~ N(x_{t-1} + (prior_t - prior_{t-1}), transition_sigma^2)``.
    Observation likelihood: ``y_t | x_t ~ N(x_t, observation_sigma^2)``.

    NaN-handling: rows with non-finite observations skip the
    likelihood update (predict-only step); particles still evolve.
    """
    T = observations.size
    out = np.full((T, 5), np.nan, dtype=np.float64)
    if T == 0:
        return out

    # Initial particles around the first finite observation (or prior).
    init_val = observations[0] if np.isfinite(observations[0]) else (
        prior[0] if np.isfinite(prior[0]) else 0.0
    )
    particles = rng.normal(
        loc=init_val, scale=max(observation_sigma, 1e-6), size=n_particles,
    )
    weights = np.full(n_particles, 1.0 / n_particles)

    for t in range(T):
        # Transition: add drift from prior (if available) + random walk noise.
        drift = 0.0
        if t > 0 and np.isfinite(prior[t]) and np.isfinite(prior[t - 1]):
            drift = float(prior[t] - prior[t - 1])
        particles = particles + drift + rng.normal(
            0.0, transition_sigma, size=n_particles,
        )

        # Likelihood update if observation is finite.
        if np.isfinite(observations[t]):
            # Gaussian log-likelihood (drop constants)
            log_lik = -0.5 * ((particles - observations[t]) / observation_sigma) ** 2
            # Numerical stability: shift by max log-lik before exp.
            log_w = np.log(weights + 1e-300) + log_lik
            log_w -= log_w.max()
            weights = np.exp(log_w)
            weights /= weights.sum() + 1e-300

            # Effective sample size -> resample if too low.
            ess = 1.0 / np.sum(weights ** 2)
            if ess < resample_threshold * n_particles:
                # Systematic resampling.
                positions = (rng.uniform(0, 1) + np.arange(n_particles)) / n_particles
                cumsum = np.cumsum(weights)
                cumsum[-1] = 1.0  # rounding guard
                idx = np.searchsorted(cumsum, positions)
                idx = np.clip(idx, 0, n_particles - 1)
                particles = particles[idx]
                weights = np.full(n_particles, 1.0 / n_particles)

        # Posterior summary: (p10, p50, p90, post_mean, neff).
        sort_idx = np.argsort(particles)
        sp = particles[sort_idx]
        sw = weights[sort_idx]
        csw = np.cumsum(sw)
        i10 = int(np.searchsorted(csw, 0.10))
        i50 = int(np.searchsorted(csw, 0.50))
        i90 = int(np.searchsorted(csw, 0.90))
        i10 = min(i10, n_particles - 1)
        i50 = min(i50, n_particles - 1)
        i90 = min(i90, n_particles - 1)
        out[t, 0] = sp[i10]
        out[t, 1] = sp[i50]
        out[t, 2] = sp[i90]
        # Weighted mean is the canonical PF posterior point estimate.
        out[t, 3] = float(np.sum(particles * weights))
        # Effective sample size (Liu 1996): 1 / sum(w^2). Diagnoses
        # particle-degeneracy independent of the resample threshold.
        out[t, 4] = 1.0 / (float(np.sum(weights ** 2)) + 1e-12)
    return out


def particle_filter_posterior(
    observations: np.ndarray,
    prior: Optional[np.ndarray] = None,
    *,
    group_ids: Optional[np.ndarray] = None,
    n_particles: int = 64,
    transition_sigma: float = 0.5,
    observation_sigma: float = 1.0,
    resample_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Bootstrap particle filter for 1-D state with Gaussian observation noise.

    Returns dict ``{"p10", "p50", "p90", "mean", "neff"}`` -- the per-
    row posterior percentile bounds + weighted mean + effective
    sample size; each value is a ``(n,)`` array.

    Parameters
    ----------
    observations
        1-D array of observations. NaN values are treated as
        "no observation this step" -> predict-only (particles drift,
        no likelihood update).
    prior
        Optional 1-D prior trajectory of the same length. Used as the
        drift term: ``x_t = x_{t-1} + (prior_t - prior_{t-1}) +
        noise``. Pass ``None`` for a pure random-walk transition
        (no drift).
    group_ids
        Per-row group identifiers. When supplied, each group runs an
        independent PF (no particle bleed across groups). When None,
        the whole array is a single group.
    n_particles
        Number of particles. 64 is a sane default for 1-D state;
        increase to 256+ for sharper bimodal posteriors.
    transition_sigma
        Std-dev of the Gaussian-random-walk transition noise. Tune
        to expected per-step state drift magnitude.
    observation_sigma
        Std-dev of the Gaussian observation likelihood. Tune to
        observation noise level.
    resample_threshold
        Resample when ESS / n_particles falls below this. Default
        0.5 = standard.
    seed
        RNG seed (per call, deterministic).
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    n = obs.size
    if prior is None:
        prior_arr = np.full(n, np.nan, dtype=np.float64)
    else:
        prior_arr = np.ascontiguousarray(prior, dtype=np.float64)
        if prior_arr.size != n:
            raise ValueError(
                f"prior length {prior_arr.size} != observations length {n}"
            )
    rng = np.random.default_rng(seed)

    out_p10 = np.full(n, np.nan, dtype=np.float64)
    out_p50 = np.full(n, np.nan, dtype=np.float64)
    out_p90 = np.full(n, np.nan, dtype=np.float64)
    out_mean = np.full(n, np.nan, dtype=np.float64)
    out_neff = np.full(n, np.nan, dtype=np.float64)

    if group_ids is None:
        res = _pf_single_segment(
            obs, prior_arr,
            n_particles=n_particles,
            transition_sigma=transition_sigma,
            observation_sigma=observation_sigma,
            resample_threshold=resample_threshold,
            rng=rng,
        )
        out_p10[:] = res[:, 0]
        out_p50[:] = res[:, 1]
        out_p90[:] = res[:, 2]
        out_mean[:] = res[:, 3]
        out_neff[:] = res[:, 4]
        return {
            "p10": out_p10, "p50": out_p50, "p90": out_p90,
            "mean": out_mean, "neff": out_neff,
        }

    from .grouped import iter_group_segments
    sort_idx, starts, ends = iter_group_segments(group_ids)
    for s, e in zip(starts, ends):
        idx_seg = sort_idx[s:e]
        seg_obs = obs[idx_seg]
        seg_prior = prior_arr[idx_seg]
        if seg_obs.size == 0:
            continue
        res = _pf_single_segment(
            seg_obs, seg_prior,
            n_particles=n_particles,
            transition_sigma=transition_sigma,
            observation_sigma=observation_sigma,
            resample_threshold=resample_threshold,
            rng=np.random.default_rng(seed),
        )
        out_p10[idx_seg] = res[:, 0]
        out_p50[idx_seg] = res[:, 1]
        out_p90[idx_seg] = res[:, 2]
        out_mean[idx_seg] = res[:, 3]
        out_neff[idx_seg] = res[:, 4]
    return {
        "p10": out_p10, "p50": out_p50, "p90": out_p90,
        "mean": out_mean, "neff": out_neff,
    }


# =============================================================================
# Kalman Filter / Smoother (1D linear-Gaussian state-space)
#
# When the system IS linear-Gaussian, KF is closed-form and 20-100x faster
# than PF at the same estimate quality. Use this as the default path; gate
# the PF only when observation likelihood is non-Gaussian or transitions
# are nonlinear.
# =============================================================================


def _kf_single_segment(
    observations: np.ndarray,
    prior_traj: np.ndarray,
    *,
    transition_sigma: float,
    observation_sigma: float,
    initial_variance: float,
) -> np.ndarray:
    """Single-segment Kalman filter.

    Returns ``(T, 5)`` array of (mean, var, innovation, innovation_var,
    log_lik) per timestep. Innovations are y_t - E[x_t | y_{1..t-1}]
    (one-step-ahead predictive residuals) -- canonical anomaly signal.
    """
    Q = transition_sigma ** 2
    R = observation_sigma ** 2
    if _NUMBA_AVAILABLE:
        # njit machine-code recurrence: ~165-290x over the Python loop,
        # bit-identical (fastmath=False). See _benchmarks/bench_kalman_filter_njit.py.
        return _kf_inner(observations, prior_traj, Q, R, float(initial_variance))
    T = observations.size
    out = np.full((T, 5), np.nan, dtype=np.float64)
    if T == 0:
        return out
    # Init from first finite observation (or 0 if none).
    init_obs = float(observations[0]) if np.isfinite(observations[0]) else 0.0
    mean = init_obs
    var = float(initial_variance)
    for t in range(T):
        # Drift from prior trajectory (if available).
        drift = 0.0
        if t > 0 and np.isfinite(prior_traj[t]) and np.isfinite(prior_traj[t - 1]):
            drift = float(prior_traj[t] - prior_traj[t - 1])
        # Predict step.
        mean_pred = mean + drift
        var_pred = var + Q
        if np.isfinite(observations[t]):
            innovation = float(observations[t] - mean_pred)
            innovation_var = var_pred + R
            K = var_pred / (innovation_var + 1e-12)
            mean = mean_pred + K * innovation
            var = (1.0 - K) * var_pred
            # log p(y_t | y_{1..t-1}) under N(mean_pred, innovation_var)
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


def kalman_filter_posterior_1d(
    observations: np.ndarray,
    prior: Optional[np.ndarray] = None,
    *,
    group_ids: Optional[np.ndarray] = None,
    transition_sigma: float = 0.5,
    observation_sigma: float = 1.0,
    initial_variance: float = 1.0,
) -> dict:
    """1D linear-Gaussian Kalman filter; default state-space path.

    Returns dict with: ``mean`` (posterior estimate), ``var``
    (posterior variance), ``innovation`` (one-step predictive
    residual), ``innovation_var`` (its variance), ``log_lik``
    (per-row log-likelihood; sum gives marginal likelihood).

    20-100x faster than the bootstrap PF for the same linear-Gaussian
    setting. Use this as the default; reach for PF only when
    observation likelihood is non-Gaussian or transitions nonlinear.

    Innovations are the canonical anomaly score: standardise
    ``innovation / sqrt(innovation_var)`` to get per-row Mahalanobis-1D
    deviation under the current model. log_lik supports model
    comparison (vary sigmas, pick the best on val log_lik sum).
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    n = obs.size
    if prior is None:
        prior_arr = np.full(n, np.nan, dtype=np.float64)
    else:
        prior_arr = np.ascontiguousarray(prior, dtype=np.float64)

    out_mean = np.full(n, np.nan, dtype=np.float64)
    out_var = np.full(n, np.nan, dtype=np.float64)
    out_innov = np.full(n, np.nan, dtype=np.float64)
    out_innov_var = np.full(n, np.nan, dtype=np.float64)
    out_log_lik = np.full(n, np.nan, dtype=np.float64)

    def _write(idx_seg, res):
        out_mean[idx_seg] = res[:, 0]
        out_var[idx_seg] = res[:, 1]
        out_innov[idx_seg] = res[:, 2]
        out_innov_var[idx_seg] = res[:, 3]
        out_log_lik[idx_seg] = res[:, 4]

    if group_ids is None:
        res = _kf_single_segment(
            obs, prior_arr,
            transition_sigma=transition_sigma,
            observation_sigma=observation_sigma,
            initial_variance=initial_variance,
        )
        _write(np.arange(n), res)
    else:
        from .grouped import iter_group_segments
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            idx_seg = sort_idx[s:e]
            if idx_seg.size == 0:
                continue
            res = _kf_single_segment(
                obs[idx_seg], prior_arr[idx_seg],
                transition_sigma=transition_sigma,
                observation_sigma=observation_sigma,
                initial_variance=initial_variance,
            )
            _write(idx_seg, res)

    return {
        "mean": out_mean,
        "var": out_var,
        "innovation": out_innov,
        "innovation_var": out_innov_var,
        "log_lik": out_log_lik,
    }


def kalman_smoother_posterior_1d(
    observations: np.ndarray,
    prior: Optional[np.ndarray] = None,
    *,
    group_ids: Optional[np.ndarray] = None,
    transition_sigma: float = 0.5,
    observation_sigma: float = 1.0,
    initial_variance: float = 1.0,
) -> dict:
    """1D RTS (Rauch-Tung-Striebel) Kalman smoother. OFFLINE-ONLY.

    !!! USES FUTURE DATA -- DO NOT APPLY ON A TEST/INFERENCE PATH !!!

    Returns ``{"mean", "var"}`` -- the smoothed (two-sided) posterior.
    Strictly better than KF where future evidence is allowed: backfills
    missing observations, denoises noisy labels, computes counterfactual
    baselines for offline analysis / target denoising / soft-labelling.

    Implementation: forward KF pass + backward RTS recursion. O(T).
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    n = obs.size
    if prior is None:
        prior_arr = np.full(n, np.nan, dtype=np.float64)
    else:
        prior_arr = np.ascontiguousarray(prior, dtype=np.float64)
    out_mean = np.full(n, np.nan, dtype=np.float64)
    out_var = np.full(n, np.nan, dtype=np.float64)
    Q = transition_sigma ** 2

    def _run(idx_seg: np.ndarray) -> None:
        seg_obs = obs[idx_seg]
        seg_prior = prior_arr[idx_seg]
        T = seg_obs.size
        if T == 0:
            return
        # Forward filter: track mean_filter / var_filter / mean_pred / var_pred.
        f_mean = np.empty(T, dtype=np.float64)
        f_var = np.empty(T, dtype=np.float64)
        p_mean = np.empty(T, dtype=np.float64)
        p_var = np.empty(T, dtype=np.float64)
        mean = float(seg_obs[0]) if np.isfinite(seg_obs[0]) else 0.0
        var = float(initial_variance)
        R = observation_sigma ** 2
        for t in range(T):
            drift = 0.0
            if t > 0 and np.isfinite(seg_prior[t]) and np.isfinite(seg_prior[t - 1]):
                drift = float(seg_prior[t] - seg_prior[t - 1])
            mean_pred = mean + drift
            var_pred = var + Q
            p_mean[t] = mean_pred
            p_var[t] = var_pred
            if np.isfinite(seg_obs[t]):
                inv = var_pred + R
                K = var_pred / (inv + 1e-12)
                mean = mean_pred + K * (seg_obs[t] - mean_pred)
                var = (1.0 - K) * var_pred
            else:
                mean = mean_pred
                var = var_pred
            f_mean[t] = mean
            f_var[t] = var
        # Backward RTS smoothing.
        s_mean = f_mean.copy()
        s_var = f_var.copy()
        for t in range(T - 2, -1, -1):
            # smoother gain
            G = f_var[t] / (p_var[t + 1] + 1e-12)
            s_mean[t] = f_mean[t] + G * (s_mean[t + 1] - p_mean[t + 1])
            s_var[t] = f_var[t] + G * G * (s_var[t + 1] - p_var[t + 1])
        out_mean[idx_seg] = s_mean
        out_var[idx_seg] = s_var

    if group_ids is None:
        _run(np.arange(n))
    else:
        from .grouped import iter_group_segments
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _run(sort_idx[s:e])

    return {"mean": out_mean, "var": out_var}


def bocpd_features(
    observations: np.ndarray,
    *,
    group_ids: Optional[np.ndarray] = None,
    hazard: float = 1.0 / 250.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    max_run_length: int = 1000,
) -> dict:
    """Adams & MacKay (2007) online Bayesian change-point detection.

    Online (causal) change-point probabilities with a Normal-Inverse-
    Gamma conjugate observation model. Closed-form, no MCMC. Returns
    dict with:
    * ``p_change`` — per-row P(run_length=0 | y_{1..t}) (probability
      that t is a change-point given everything observed so far)
    * ``expected_run_length`` — mean run length under posterior
    * ``max_run_length`` — MAP run length

    Use cases: regime detection in finance (vol regime switches),
    epi (R_t breakpoints), sensor failures, fraud onset.
    `expected_run_length` is a natural "time-since-last-shift" feature.

    ``max_run_length`` caps the run-length posterior vector. The Adams-MacKay recursion otherwise grows the run-length
    distribution by one element per observation -- O(T) state and O(T^2) total work -- which is unbounded on long /
    streaming inputs. The cap truncates the (lowest-probability) tail of large run lengths each step and renormalises
    the surviving head, bounding state to O(max_run_length) and per-step work likewise. The default 1000 is far above
    the run lengths that carry meaningful posterior mass under realistic hazards (e.g. hazard=1/250 -> typical runs in
    the hundreds), so for typical streams the output is unchanged vs uncapped; pass ``max_run_length<=0`` to disable
    the cap entirely (legacy uncapped behaviour). Note: this caps the run-length VECTOR; the returned ``max_run_length``
    dict key is the per-row MAP run length and is unrelated to this argument.
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    n = obs.size
    out_p_change = np.full(n, np.nan, dtype=np.float64)
    out_expected_rl = np.full(n, np.nan, dtype=np.float64)
    out_max_rl = np.full(n, np.nan, dtype=np.float64)

    def _run(idx_seg: np.ndarray) -> None:
        seg = np.ascontiguousarray(obs[idx_seg], dtype=np.float64)
        T = seg.size
        if T == 0:
            return
        if _NUMBA_AVAILABLE:
            seg_p = np.empty(T, dtype=np.float64)
            seg_ex = np.empty(T, dtype=np.float64)
            seg_max = np.empty(T, dtype=np.float64)
            _bocpd_inner(seg, hazard, mu0, kappa0, alpha0, beta0,
                         seg_p, seg_ex, seg_max, max_run_length)
            out_p_change[idx_seg] = seg_p
            out_expected_rl[idx_seg] = seg_ex
            out_max_rl[idx_seg] = seg_max
            return
        # NumPy fallback (numba unavailable).
        mu = np.array([mu0], dtype=np.float64)
        kappa = np.array([kappa0], dtype=np.float64)
        alpha = np.array([alpha0], dtype=np.float64)
        beta = np.array([beta0], dtype=np.float64)
        rl_probs = np.array([1.0], dtype=np.float64)
        for t in range(T):
            x = float(seg[t])
            if not np.isfinite(x):
                rl_probs_new = np.concatenate(([0.0], rl_probs * (1.0 - hazard)))
                rl_probs_new[0] = (rl_probs * hazard).sum()
                rl_probs_new = rl_probs_new / (rl_probs_new.sum() + 1e-300)
                rl_probs = rl_probs_new
                mu = np.concatenate(([mu0], mu))
                kappa = np.concatenate(([kappa0], kappa))
                alpha = np.concatenate(([alpha0], alpha))
                beta = np.concatenate(([beta0], beta))
                if max_run_length > 0 and rl_probs.size > max_run_length:
                    rl_probs = rl_probs[:max_run_length]
                    rl_probs = rl_probs / (rl_probs.sum() + 1e-300)
                    mu = mu[:max_run_length]; kappa = kappa[:max_run_length]
                    alpha = alpha[:max_run_length]; beta = beta[:max_run_length]
                out_p_change[idx_seg[t]] = rl_probs[0]
                rl_idx = np.arange(rl_probs.size, dtype=np.float64)
                out_expected_rl[idx_seg[t]] = float((rl_idx * rl_probs).sum())
                out_max_rl[idx_seg[t]] = float(np.argmax(rl_probs))
                continue
            df = 2.0 * alpha
            scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
            half_df = 0.5 * df
            z2 = (x - mu) ** 2 / scale_sq
            log_pred = (
                _gammaln(half_df + 0.5)
                - _gammaln(half_df)
                - 0.5 * np.log(np.pi * df * scale_sq)
                - (half_df + 0.5) * np.log1p(z2 / df)
            )
            pred = np.exp(log_pred - log_pred.max())
            growth = rl_probs * pred * (1.0 - hazard)
            cp = (rl_probs * pred * hazard).sum()
            rl_probs_new = np.concatenate(([cp], growth))
            rl_probs_new = rl_probs_new / (rl_probs_new.sum() + 1e-300)
            mu_new = np.concatenate(([mu0], (kappa * mu + x) / (kappa + 1.0)))
            kappa_new = np.concatenate(([kappa0], kappa + 1.0))
            alpha_new = np.concatenate(([alpha0], alpha + 0.5))
            beta_new = np.concatenate((
                [beta0],
                beta + 0.5 * (kappa * (x - mu) ** 2) / (kappa + 1.0),
            ))
            mu = mu_new; kappa = kappa_new
            alpha = alpha_new; beta = beta_new
            rl_probs = rl_probs_new
            if max_run_length > 0 and rl_probs.size > max_run_length:
                rl_probs = rl_probs[:max_run_length]
                rl_probs = rl_probs / (rl_probs.sum() + 1e-300)
                mu = mu[:max_run_length]; kappa = kappa[:max_run_length]
                alpha = alpha[:max_run_length]; beta = beta[:max_run_length]
            out_p_change[idx_seg[t]] = float(rl_probs[0])
            rl_idx = np.arange(rl_probs.size, dtype=np.float64)
            out_expected_rl[idx_seg[t]] = float((rl_idx * rl_probs).sum())
            out_max_rl[idx_seg[t]] = float(np.argmax(rl_probs))

    if group_ids is None:
        _run(np.arange(n))
    elif _NUMBA_AVAILABLE:
        # Group-parallel njit path: each group is an independent recursion,
        # so dispatch serial vs prange-over-groups by group count + size.
        from .grouped import iter_group_segments
        from ._recursion_dispatch import dispatch_recursion_backend
        sort_idx, starts, ends = iter_group_segments(group_ids)
        obs_sorted = np.ascontiguousarray(obs[sort_idx])
        p_s = np.empty(n, dtype=np.float64)
        ex_s = np.empty(n, dtype=np.float64)
        max_s = np.empty(n, dtype=np.float64)
        backend = dispatch_recursion_backend("fe_bocpd", n, int(starts.size))
        if backend == "parallel":
            _bocpd_groups_parallel(obs_sorted, starts, ends, hazard,
                                   mu0, kappa0, alpha0, beta0, p_s, ex_s, max_s, max_run_length)
        else:
            _bocpd_groups_serial(obs_sorted, starts, ends, hazard,
                                 mu0, kappa0, alpha0, beta0, p_s, ex_s, max_s, max_run_length)
        out_p_change[sort_idx] = p_s
        out_expected_rl[sort_idx] = ex_s
        out_max_rl[sort_idx] = max_s
    else:
        from .grouped import iter_group_segments
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _run(sort_idx[s:e])

    return {
        "p_change": out_p_change,
        "expected_run_length": out_expected_rl,
        "max_run_length": out_max_rl,
    }


def online_bayesian_linear_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    group_ids: Optional[np.ndarray] = None,
    prior_precision: float = 1.0,
    noise_sigma: float = 1.0,
) -> dict:
    """Recursive Bayesian linear regression (NIG conjugate).

    Streaming online updates of the posterior over slope+intercept of
    ``y ~ X @ beta + noise``. Closed-form (no MCMC), O(k^2) per step.
    Returns dict with:
    * ``predictive_mean`` — one-step predictive mean ``X[t] @ E[beta]``
    * ``predictive_var`` — predictive variance (depends on row, not const)
    * ``log_marginal_lik`` — incremental log-evidence per row

    Use cases: finance beta tracking (slope drift over time), dose-
    response slope, ad-spend elasticity by day, sensor calibration
    coefficients. Distinct from KF on (k=1, X=ones): general case with
    multiple features and slope-uncertainty as a per-row feature.
    """
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.size:
        raise ValueError(
            f"X must be 2-D (n, k) with X.shape[0]=={y_arr.size}; got {X_arr.shape}"
        )
    n, k = X_arr.shape
    out_pred_mean = np.full(n, np.nan, dtype=np.float64)
    out_pred_var = np.full(n, np.nan, dtype=np.float64)
    out_log_marg = np.full(n, np.nan, dtype=np.float64)

    def _run(idx_seg: np.ndarray) -> None:
        idx = np.ascontiguousarray(idx_seg, dtype=np.int64)
        if idx.size == 0:
            return
        if _NUMBA_AVAILABLE:
            y_sub = np.ascontiguousarray(y_arr[idx])
            X_sub = np.ascontiguousarray(X_arr[idx])
            pm = np.empty(idx.size, dtype=np.float64)
            pv = np.empty(idx.size, dtype=np.float64)
            lm = np.full(idx.size, np.nan, dtype=np.float64)
            _oblr_inner(y_sub, X_sub, prior_precision, noise_sigma, pm, pv, lm)
            out_pred_mean[idx] = pm
            out_pred_var[idx] = pv
            out_log_marg[idx] = lm
            return
        # Track covariance Sigma + Sherman-Morrison update: O(k^2) per step.
        mu = np.zeros(k, dtype=np.float64)
        Sigma = np.eye(k, dtype=np.float64) / prior_precision
        noise_var = noise_sigma ** 2
        for t in idx_seg:
            x_t = X_arr[t]
            y_t = float(y_arr[t])
            Sx = Sigma @ x_t
            pred_mean = float(x_t @ mu)
            pred_var = float(x_t @ Sx) + noise_var
            out_pred_mean[t] = pred_mean
            out_pred_var[t] = pred_var
            if np.isfinite(y_t):
                out_log_marg[t] = -0.5 * (
                    math.log(2.0 * math.pi * pred_var)
                    + (y_t - pred_mean) ** 2 / pred_var
                )
                innovation = y_t - pred_mean
                K = Sx / pred_var
                mu = mu + K * innovation
                Sigma = Sigma - np.outer(K, Sx)

    if group_ids is None:
        _run(np.arange(n))
    elif _NUMBA_AVAILABLE:
        from .grouped import iter_group_segments
        from ._recursion_dispatch import dispatch_recursion_backend
        sort_idx, starts, ends = iter_group_segments(group_ids)
        y_sorted = np.ascontiguousarray(y_arr[sort_idx])
        X_sorted = np.ascontiguousarray(X_arr[sort_idx])
        pm = np.empty(n, dtype=np.float64)
        pv = np.empty(n, dtype=np.float64)
        lm = np.full(n, np.nan, dtype=np.float64)
        backend = dispatch_recursion_backend("fe_oblr", n, int(starts.size))
        if backend == "parallel":
            _oblr_groups_parallel(y_sorted, X_sorted, starts, ends,
                                  prior_precision, noise_sigma, pm, pv, lm)
        else:
            _oblr_groups_serial(y_sorted, X_sorted, starts, ends,
                                prior_precision, noise_sigma, pm, pv, lm)
        out_pred_mean[sort_idx] = pm
        out_pred_var[sort_idx] = pv
        out_log_marg[sort_idx] = lm
    else:
        from .grouped import iter_group_segments
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _run(sort_idx[s:e])

    return {
        "predictive_mean": out_pred_mean,
        "predictive_var": out_pred_var,
        "log_marginal_lik": out_log_marg,
    }
