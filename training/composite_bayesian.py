"""OPEN-B helper: bayesian_alpha_fit bootstrap posterior for linear_residual (alpha, beta) coefficients. Returns posterior mean / std / credible interval per coefficient + raw bootstrap samples. Lazy-imports ``_linear_residual_fit`` from composite.py."""


from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# bayesian_alpha_fit (R10c brainstorm round-2 extension B; posterior for linear_residual alpha).
#
# Bootstrap-based posterior for the (alpha, beta) of ``linear_residual``. Returns posterior mean / std / credible interval per coefficient plus the raw bootstrap samples so downstream code can compute custom percentiles. Useful for:
# - Uncertainty quantification in reports (``alpha = 1.04 +/- 0.02 [95% CI 1.01, 1.07]``).
# - Empirical-Bayes shrinkage: a small alpha CI means strong evidence, a wide CI suggests shrinkage to a global prior.
# - Production diagnostic: if the posterior std is large relative to the point estimate, the OLS fit is fragile and the composite spec should be down-weighted in the ensemble.
#
# On large n (e.g. 4M-row TVT) the bootstrap collapses to a delta function (very tight posterior) so the practical value is small. On small-n targets (50K rows or per-group fits with K=10) the posterior carries real information.
# ----------------------------------------------------------------------

_BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP: int = 200
_BAYESIAN_ALPHA_DEFAULT_CI_LEVEL: float = 0.95


def bayesian_alpha_fit(
    y: np.ndarray,
    base: np.ndarray,
    *,
    n_bootstrap: int = _BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP,
    ci_level: float = _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
    random_state: int = 42,
    subsample_n: Optional[int] = None,
) -> Dict[str, Any]:
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
    from .composite import _linear_residual_fit
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
