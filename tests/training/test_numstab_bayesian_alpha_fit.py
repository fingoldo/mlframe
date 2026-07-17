"""Numerical-stability regression test for ``bayesian_alpha_fit``.

Pre-fix the OLS step used ``np.linalg.inv(XtX)``; a near-constant ``base`` makes ``XtX``
ill-conditioned (cond ~1e16) WITHOUT being exactly singular, so ``inv`` did not raise the
``LinAlgError`` that routes to the degenerate fallback. It instead returned a (X'X)^-1 with
~1e13 diagonal entries, exploding the posterior variances and yielding a meaningless
alpha_mean. The lstsq pseudo-inverse keeps the inverse bounded.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery.bayesian import bayesian_alpha_fit


def test_bayesian_alpha_fit_near_singular_base_bounded_posterior():
    rng = np.random.default_rng(0)
    n = 200
    base = np.ones(n) + 1e-11 * rng.standard_normal(n)  # near-constant -> XtX cond ~1e16
    y = 2.0 * base + 0.5 + 0.01 * rng.standard_normal(n)

    out = bayesian_alpha_fit(y, base, random_state=0)

    assert np.isfinite(out["alpha_mean"]), "alpha_mean must be finite on near-singular design"
    # Pre-fix the exploded (X'X)^-1 drove |alpha_mean| to ~24 (true alpha is 2) and the
    # posterior std to absurd magnitudes; bound both well below the pre-fix blow-up.
    assert abs(out["alpha_mean"]) < 1e3, f"alpha_mean exploded: {out['alpha_mean']}"
    if np.isfinite(out["alpha_std"]):
        assert out["alpha_std"] < 1e3, f"alpha_std exploded: {out['alpha_std']}"
