"""One BayesianGaussianMixture fit-and-sample, shared by the two BGMM oversampling transformers.

``bgmm_multiscale`` and ``bgmm_virtual`` each carried their own ``_fit_bgmm_and_sample``. Compared on the
parse tree, the two bodies were identical apart from the log label -- so they were numerically the same
function, and a change to the prior, ``reg_covar``, ``max_iter`` or the bootstrap fallback would have
reached whichever module the author happened to open.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def fit_bgmm_and_sample(X_minority: np.ndarray, n_synthetic: int, n_components: int, seed: int, *, caller: str) -> np.ndarray:
    """Fit a BayesianGaussianMixture on the minority rows and draw ``n_synthetic`` virtual samples.

    Bootstraps instead when there are too few minority rows to fit ``n_components``, and again if the fit
    itself raises -- a mixture that will not converge must degrade to resampling rather than fail the
    transform. ``caller`` only names the module in that fallback's log line.
    """
    from sklearn.mixture import BayesianGaussianMixture

    n_min = X_minority.shape[0]
    if n_min < n_components + 1:
        return np.asarray(X_minority[np.random.default_rng(seed).integers(0, n_min, size=n_synthetic)].copy().astype(np.float32))
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,
        weight_concentration_prior_type="dirichlet_process",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bgm.fit(X_minority)
            samples, _ = bgm.sample(n_synthetic)
        except Exception as exc:
            logger.info("%s: BGM fit failed at K=%d (%s); falling back to bootstrap.", caller, n_components, exc)
            rng = np.random.default_rng(seed)
            samples = X_minority[rng.integers(0, n_min, size=n_synthetic)]
    return np.asarray(samples.astype(np.float32))
