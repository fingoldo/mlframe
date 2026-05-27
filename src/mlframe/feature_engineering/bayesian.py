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
    T = observations.size
    out = np.full((T, 5), np.nan, dtype=np.float64)
    if T == 0:
        return out
    Q = transition_sigma ** 2
    R = observation_sigma ** 2
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
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    n = obs.size
    out_p_change = np.full(n, np.nan, dtype=np.float64)
    out_expected_rl = np.full(n, np.nan, dtype=np.float64)
    out_max_rl = np.full(n, np.nan, dtype=np.float64)

    def _run(idx_seg: np.ndarray) -> None:
        seg = obs[idx_seg]
        T = seg.size
        if T == 0:
            return
        # Cap run length to T (max possible).
        # Maintain (mu, kappa, alpha, beta) suff stats per run length.
        mu = np.array([mu0], dtype=np.float64)
        kappa = np.array([kappa0], dtype=np.float64)
        alpha = np.array([alpha0], dtype=np.float64)
        beta = np.array([beta0], dtype=np.float64)
        rl_probs = np.array([1.0], dtype=np.float64)  # P(r_0 = 0) = 1
        for t in range(T):
            x = float(seg[t])
            if not np.isfinite(x):
                # Predict-only: propagate growth without update.
                rl_probs_new = np.concatenate(([0.0], rl_probs * (1.0 - hazard)))
                rl_probs_new[0] = (rl_probs * hazard).sum()
                rl_probs_new = rl_probs_new / (rl_probs_new.sum() + 1e-300)
                rl_probs = rl_probs_new
                # Suff stats keep same (no observation).
                mu = np.concatenate(([mu0], mu))
                kappa = np.concatenate(([kappa0], kappa))
                alpha = np.concatenate(([alpha0], alpha))
                beta = np.concatenate(([beta0], beta))
                out_p_change[idx_seg[t]] = rl_probs[0]
                rl_idx = np.arange(rl_probs.size, dtype=np.float64)
                out_expected_rl[idx_seg[t]] = float((rl_idx * rl_probs).sum())
                out_max_rl[idx_seg[t]] = float(np.argmax(rl_probs))
                continue
            # Predictive: Student-t with mean=mu, scale = sqrt(beta * (kappa+1) / (alpha * kappa)), df=2*alpha
            # Use log-pdf for stability.
            df = 2.0 * alpha
            scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
            log_pred = (
                -0.5 * np.log(np.pi * df * scale_sq)
                + np.log(np.exp(  # only for ratio form; compute via lgamma
                    0.0
                ))  # placeholder; switch to full log-form below
            )
            # Use scipy.stats.t.logpdf for accuracy.
            from scipy.stats import t as _student_t
            log_pred = _student_t.logpdf(
                x, df=df, loc=mu, scale=np.sqrt(scale_sq),
            )
            pred = np.exp(log_pred - log_pred.max())
            # Growth probabilities: r_t = r_{t-1} + 1
            growth = rl_probs * pred * (1.0 - hazard)
            # Change-point probability: r_t = 0
            cp = (rl_probs * pred * hazard).sum()
            rl_probs_new = np.concatenate(([cp], growth))
            rl_probs_new = rl_probs_new / (rl_probs_new.sum() + 1e-300)
            # Update suff stats: prepend the prior, increment existing.
            mu_new = np.concatenate(([mu0], (kappa * mu + x) / (kappa + 1.0)))
            kappa_new = np.concatenate(([kappa0], kappa + 1.0))
            alpha_new = np.concatenate(([alpha0], alpha + 0.5))
            beta_new = np.concatenate((
                [beta0],
                beta + 0.5 * (kappa * (x - mu) ** 2) / (kappa + 1.0),
            ))
            mu = mu_new
            kappa = kappa_new
            alpha = alpha_new
            beta = beta_new
            rl_probs = rl_probs_new
            out_p_change[idx_seg[t]] = float(rl_probs[0])
            rl_idx = np.arange(rl_probs.size, dtype=np.float64)
            out_expected_rl[idx_seg[t]] = float((rl_idx * rl_probs).sum())
            out_max_rl[idx_seg[t]] = float(np.argmax(rl_probs))

    if group_ids is None:
        _run(np.arange(n))
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
        # State: posterior over beta is N(mu, Sigma) with precision
        # matrix Lambda = Sigma^-1.
        mu = np.zeros(k, dtype=np.float64)
        Lambda = prior_precision * np.eye(k, dtype=np.float64)
        noise_var = noise_sigma ** 2
        for t in idx_seg:
            x_t = X_arr[t]
            y_t = float(y_arr[t])
            # Predictive: y_t | data ~ N(x^T mu, x^T Sigma x + noise_var)
            Sigma = np.linalg.solve(Lambda, np.eye(k))
            pred_mean = float(x_t @ mu)
            pred_var = float(x_t @ Sigma @ x_t) + noise_var
            out_pred_mean[t] = pred_mean
            out_pred_var[t] = pred_var
            if np.isfinite(y_t):
                # log marginal lik under N(pred_mean, pred_var)
                out_log_marg[t] = -0.5 * (
                    math.log(2.0 * math.pi * pred_var)
                    + (y_t - pred_mean) ** 2 / pred_var
                )
                # Bayesian update of precision + mean.
                Lambda_new = Lambda + np.outer(x_t, x_t) / noise_var
                # mu_new = Lambda_new^-1 (Lambda mu + x_t y_t / noise_var)
                rhs = Lambda @ mu + x_t * (y_t / noise_var)
                mu = np.linalg.solve(Lambda_new, rhs)
                Lambda = Lambda_new

    if group_ids is None:
        _run(np.arange(n))
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
