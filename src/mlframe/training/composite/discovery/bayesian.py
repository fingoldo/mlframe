"""Bayesian posterior for linear_residual (alpha, beta) coefficients.

Two functions ship together:

- :func:`bayesian_alpha_fit` (default) implements a proper conjugate
  Normal-Inverse-Gamma posterior on the linear model
  ``y = alpha * base + beta + eps,  eps ~ N(0, sigma^2)`` with a flat
  (non-informative) prior. Returns posterior mean / std / credible
  interval for alpha and beta plus posterior samples drawn from the
  multivariate-t marginal.
- :func:`bayesian_alpha_fit_bootstrap` is the legacy resample-with-
  replacement bootstrap posterior. Kept as an opt-in fallback for
  callers that need the non-parametric flavour (e.g. heavily non-
  Gaussian residuals where the t-marginal is misleading). The two
  functions share a return-dict shape so swapping is a one-line change.

Lazy-imports ``_linear_residual_fit`` from composite.py to break the
import cycle.
"""


from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Posterior summary for the (alpha, beta) of ``linear_residual``. Two
# variants live in this module:
# 1. ``bayesian_alpha_fit``: real Bayesian posterior with the conjugate
#    Normal-Inverse-Gamma prior. Closed-form on n; the marginal of
#    (alpha, beta) is a multivariate-t with v = n-2 degrees of freedom.
#    On large n the t collapses to a normal, so the result coincides
#    with the OLS standard-error interval.
# 2. ``bayesian_alpha_fit_bootstrap``: empirical bootstrap (legacy
#    implementation). Useful for residual distributions where the
#    Gaussian-eps assumption is dubious.


_BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP: int = 200
_BAYESIAN_ALPHA_DEFAULT_CI_LEVEL: float = 0.95
_BAYESIAN_ALPHA_POSTERIOR_N_SAMPLES: int = 500


def bayesian_alpha_fit(
    y: np.ndarray,
    base: np.ndarray,
    *,
    ci_level: float = _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
    random_state: int = 42,
    n_samples: int = _BAYESIAN_ALPHA_POSTERIOR_N_SAMPLES,
) -> dict[str, Any]:
    """Conjugate Normal-Inverse-Gamma posterior for (alpha, beta).

    Model: ``y = alpha * base + beta + eps,  eps ~ N(0, sigma^2)``.
    With a flat improper prior ``p(alpha, beta, sigma^2) propto 1/sigma^2``,
    the posterior is the standard Bayesian linear regression result:

    - sigma^2 | y ~ Inverse-Gamma((n-2)/2, SSR/2)  where SSR is the
      sum-of-squared residuals at the OLS point estimate.
    - (alpha, beta) | sigma^2, y ~ Normal(OLS estimate, sigma^2 * (X'X)^-1).
    - Marginalising sigma^2 out gives a multivariate-t with v = n-2 df,
      location = OLS estimate, scale = sigma_hat^2 * (X'X)^-1.

    Posterior summary uses the analytic mean / std of the multivariate-t
    (closed form for v > 2). Samples are drawn for downstream consumers
    that want custom percentiles; we sample sigma^2 from its Inverse-Gamma
    posterior and then (alpha, beta) | sigma^2 from the corresponding
    Gaussian.

    Returns the same dict shape as the legacy bootstrap variant so
    callers can swap impls without code changes.
    """
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n = y_f.size
    if n < 4:
        # Degenerate: no posterior to compute. Mirror the legacy degenerate
        # contract so swapping implementations is safe.
        from .. import _linear_residual_fit
        params = _linear_residual_fit(y_f, base_f)
        return {
            "alpha_mean": params["alpha"],
            "alpha_std": float("nan"),
            "alpha_ci_low": float("nan"),
            "alpha_ci_high": float("nan"),
            "beta_mean": params["beta"],
            "beta_std": float("nan"),
            "beta_ci_low": float("nan"),
            "beta_ci_high": float("nan"),
            "alpha_samples": np.array([params["alpha"]], dtype=np.float64),
            "beta_samples": np.array([params["beta"]], dtype=np.float64),
            "n_bootstrap": 0,
            "n_samples": 0,
            "ci_level": float(ci_level),
            "posterior_kind": "conjugate_normal_inverse_gamma",
            "degrees_of_freedom": float(max(n - 2, 0)),
        }
    # Design matrix [base, 1] -- same convention as ``_linear_residual_fit``.
    X = np.column_stack([base_f, np.ones(n, dtype=np.float64)])
    # OLS point estimate.
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Degenerate design (constant base). Return a wide-but-finite posterior.
        from .. import _linear_residual_fit
        params = _linear_residual_fit(y_f, base_f)
        return {
            "alpha_mean": params["alpha"],
            "alpha_std": float("inf"),
            "alpha_ci_low": float("-inf"),
            "alpha_ci_high": float("inf"),
            "beta_mean": params["beta"],
            "beta_std": float("inf"),
            "beta_ci_low": float("-inf"),
            "beta_ci_high": float("inf"),
            "alpha_samples": np.array([params["alpha"]], dtype=np.float64),
            "beta_samples": np.array([params["beta"]], dtype=np.float64),
            "n_bootstrap": 0,
            "n_samples": 0,
            "ci_level": float(ci_level),
            "posterior_kind": "conjugate_normal_inverse_gamma_degenerate",
            "degrees_of_freedom": float(max(n - 2, 0)),
        }
    coef = XtX_inv @ (X.T @ y_f)
    alpha_hat, beta_hat = float(coef[0]), float(coef[1])
    residuals = y_f - X @ coef
    ssr = float(residuals @ residuals)
    v = n - 2  # degrees of freedom
    # Posterior on sigma^2 is Inverse-Gamma(v/2, ssr/2); its MEAN is
    # ssr / (v - 2), MODE is ssr / (v + 2). We use s2_hat = ssr / v
    # (the unbiased OLS estimator) as the location for the t-marginal,
    # consistent with the standard ``Bayesian linear regression with
    # Jeffreys prior`` derivation.
    s2_hat = ssr / v if v > 0 else float("nan")
    # Covariance of (alpha, beta) | y under the multivariate-t is
    # s2_hat * (X'X)^-1 * v / (v - 2)  for v > 2. We expose std of the
    # marginal t directly: std_alpha = sqrt(s2_hat * XtX_inv[0,0] * v/(v-2)).
    if v > 2:
        var_factor = v / (v - 2)
    else:
        # C-Low-11: surface degenerate-posterior at WARN. v=n-2; v<=2 (i.e. n<=4) makes the
        # marginal t-variance factor v/(v-2) undefined. Operators feeding 4-row data previously
        # got a valid alpha_mean and a silent NaN CI -- this WARN lets them grep for the case.
        logger.warning(
            "[bayesian_alpha_fit] degenerate posterior: n=%d -> v=n-2=%d <= 2; "
            "variance factor v/(v-2) undefined, alpha_std / beta_std / CIs will be NaN. "
            "Pass at least 5 observations for a finite posterior variance.",
            n, v,
        )
        var_factor = float("nan")
    alpha_var = s2_hat * float(XtX_inv[0, 0]) * var_factor
    beta_var = s2_hat * float(XtX_inv[1, 1]) * var_factor
    alpha_std = float(np.sqrt(max(alpha_var, 0.0))) if np.isfinite(alpha_var) else float("nan")
    beta_std = float(np.sqrt(max(beta_var, 0.0))) if np.isfinite(beta_var) else float("nan")
    # Credible interval from the Student-t quantile.
    half_tail = (1.0 - float(ci_level)) / 2.0
    try:
        from scipy.stats import t as _t_dist
        t_q = float(_t_dist.ppf(1.0 - half_tail, df=max(v, 1)))
    except ImportError:
        # Fallback to normal quantile when scipy.stats is unavailable;
        # underestimates tails for small v but never crashes.
        try:
            from scipy.stats import norm as _norm
            t_q = float(_norm.ppf(1.0 - half_tail))
        except ImportError:
            t_q = 1.96  # 95% normal approximation
    sqrt_s2 = float(np.sqrt(max(s2_hat, 0.0)))
    alpha_t_scale = sqrt_s2 * float(np.sqrt(max(XtX_inv[0, 0], 0.0)))
    beta_t_scale = sqrt_s2 * float(np.sqrt(max(XtX_inv[1, 1], 0.0)))
    alpha_ci_low = alpha_hat - t_q * alpha_t_scale
    alpha_ci_high = alpha_hat + t_q * alpha_t_scale
    beta_ci_low = beta_hat - t_q * beta_t_scale
    beta_ci_high = beta_hat + t_q * beta_t_scale
    # Draw posterior samples via the hierarchical recipe so downstream
    # consumers can recover custom percentiles or build joint plots.
    rng = np.random.default_rng(random_state)
    n_draws = max(int(n_samples), 1)
    # sigma^2 ~ Inverse-Gamma(v/2, ssr/2) <=> 1/sigma^2 ~ Gamma(v/2, scale=2/ssr).
    if v > 0 and ssr > 0:
        inv_sigma2 = rng.gamma(shape=v / 2.0, scale=2.0 / ssr, size=n_draws)
        sigma2_samples = 1.0 / inv_sigma2
    else:
        sigma2_samples = np.full(n_draws, max(s2_hat, 0.0))
    # (alpha, beta) | sigma^2 ~ N(coef, sigma^2 * XtX_inv); draw via Cholesky.
    try:
        L = np.linalg.cholesky(XtX_inv)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.clip(np.diag(XtX_inv), 0.0, None)))
    z = rng.standard_normal(size=(n_draws, 2))
    scaled = z @ L.T  # (n_draws, 2)
    scale = np.sqrt(sigma2_samples)[:, None]
    samples = coef[None, :] + scale * scaled  # (n_draws, 2)
    alpha_samples = samples[:, 0]
    beta_samples = samples[:, 1]
    return {
        "alpha_mean": alpha_hat,
        "alpha_std": alpha_std,
        "alpha_ci_low": float(alpha_ci_low),
        "alpha_ci_high": float(alpha_ci_high),
        "beta_mean": beta_hat,
        "beta_std": beta_std,
        "beta_ci_low": float(beta_ci_low),
        "beta_ci_high": float(beta_ci_high),
        "alpha_samples": alpha_samples,
        "beta_samples": beta_samples,
        "n_bootstrap": 0,
        "n_samples": int(n_draws),
        "ci_level": float(ci_level),
        "posterior_kind": "conjugate_normal_inverse_gamma",
        "degrees_of_freedom": float(v),
        "sigma2_posterior_mean": float(ssr / max(v - 2, 1)) if v > 2 else float("nan"),
    }


def bayesian_alpha_fit_bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    *,
    n_bootstrap: int = _BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP,
    ci_level: float = _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
    random_state: int = 42,
    subsample_n: int | None = None,
) -> dict[str, Any]:
    """Bootstrap posterior for linear_residual (alpha, beta).

    Parameters
    ----------
    y, base
        Training arrays (1-D). Caller must filter to the valid domain.
    n_bootstrap
        Number of resamples-with-replacement (default 200; balance between posterior tightness and compute).
    ci_level
        Two-sided credible interval level (default 0.95 -> 2.5 / 97.5 percentiles).
    random_state
        RNG seed; reproducible across runs.
    subsample_n
        When provided, each bootstrap draw uses a subsample of this size (not the full n) to keep compute bounded on huge datasets. ``None`` uses full n.

    Returns
    -------
    Posterior summary dict with keys:
    - ``alpha_mean`` / ``alpha_std`` / ``alpha_ci_low`` / ``alpha_ci_high``: posterior moments + credible interval.
    - ``beta_mean`` / ``beta_std`` / ``beta_ci_low`` / ``beta_ci_high``: same for the intercept.
    - ``alpha_samples`` / ``beta_samples``: 1-D ndarrays of length ``n_bootstrap`` for downstream custom percentiles.
    - ``n_bootstrap``: int (recorded for reproducibility).
    - ``ci_level``: float.
    """
    # Lazy-import composite-internal helper to break the import cycle.
    from .. import _linear_residual_fit
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n = y_f.size
    if n < 4:
        # Degenerate: no posterior to compute.
        params = _linear_residual_fit(y_f, base_f)
        return {
            "alpha_mean": params["alpha"],
            "alpha_std": float("nan"),
            "alpha_ci_low": float("nan"),
            "alpha_ci_high": float("nan"),
            "beta_mean": params["beta"],
            "beta_std": float("nan"),
            "beta_ci_low": float("nan"),
            "beta_ci_high": float("nan"),
            "alpha_samples": np.array([params["alpha"]], dtype=np.float64),
            "beta_samples": np.array([params["beta"]], dtype=np.float64),
            "n_bootstrap": 0,
            "ci_level": float(ci_level),
        }
    rng = np.random.default_rng(random_state)
    sample_size = int(subsample_n) if subsample_n is not None else n
    sample_size = max(2, min(sample_size, n))
    alphas = np.empty(n_bootstrap, dtype=np.float64)
    betas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=sample_size)
        params = _linear_residual_fit(y_f[idx], base_f[idx])
        alphas[b] = params["alpha"]
        betas[b] = params["beta"]
    # Posterior summary -- means / stds / percentiles.
    half_tail = (1.0 - float(ci_level)) / 2.0
    q_lo, q_hi = 100.0 * half_tail, 100.0 * (1.0 - half_tail)
    return {
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas, ddof=1)) if n_bootstrap > 1 else float("nan"),
        "alpha_ci_low": float(np.percentile(alphas, q_lo)),
        "alpha_ci_high": float(np.percentile(alphas, q_hi)),
        "beta_mean": float(np.mean(betas)),
        "beta_std": float(np.std(betas, ddof=1)) if n_bootstrap > 1 else float("nan"),
        "beta_ci_low": float(np.percentile(betas, q_lo)),
        "beta_ci_high": float(np.percentile(betas, q_hi)),
        "alpha_samples": alphas,
        "beta_samples": betas,
        "n_bootstrap": int(n_bootstrap),
        "ci_level": float(ci_level),
    }
