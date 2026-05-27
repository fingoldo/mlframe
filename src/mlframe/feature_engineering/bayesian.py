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
]

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
