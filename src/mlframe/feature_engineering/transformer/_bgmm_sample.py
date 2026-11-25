"""One BayesianGaussianMixture fit-and-sample, shared by the two BGMM oversampling transformers.

``bgmm_multiscale`` and ``bgmm_virtual`` each carried their own ``_fit_bgmm_and_sample``. Compared on the
parse tree, the two bodies were identical apart from the log label -- so they were numerically the same
function, and a change to the prior, ``reg_covar``, ``max_iter`` or the bootstrap fallback would have
reached whichever module the author happened to open.
"""

from __future__ import annotations

import logging
import warnings

from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def fit_bgmm_and_sample(X_minority: np.ndarray, n_synthetic: int, n_components: int, seed: int, *, caller: str, return_source: bool = False) -> Any:
    """Fit a BayesianGaussianMixture on the minority rows and draw ``n_synthetic`` virtual samples.

    Bootstraps instead when there are too few minority rows to fit ``n_components``, and again if the fit
    itself raises -- a mixture that will not converge must degrade to resampling rather than fail the
    transform. ``caller`` only names the module in that fallback's log line.

    With ``return_source=True`` the result is ``(samples, source)`` where ``source`` is ``"bgmm"`` for a real mixture sample and ``"bootstrap"``
    for a resample, so a caller can tell them apart (audit 08-F4). The default keeps the array-only return.
    """
    from sklearn.mixture import BayesianGaussianMixture

    n_min = X_minority.shape[0]
    if n_min == 0:  # nothing to bootstrap from: name the cause rather than fail inside numpy's integers(0, 0) (08-F18)
        raise ValueError(f"fit_bgmm_and_sample ({caller}): no minority rows to sample from.")
    if n_min < n_components + 1:
        boot = np.asarray(X_minority[np.random.default_rng(seed).integers(0, n_min, size=n_synthetic)].copy().astype(np.float32))
        return (boot, "bootstrap") if return_source else boot
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,
        weight_concentration_prior_type="dirichlet_process",
    )
    source = "bgmm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        try:
            bgm.fit(X_minority)
            samples, _ = bgm.sample(n_synthetic)
        except Exception as exc:
            # WARNING, not INFO: a bootstrap fallback is not a BGM sample and must be visible in normal runs (08-F4).
            logger.warning("%s: BGM fit failed at K=%d (%s); falling back to bootstrap.", caller, n_components, exc)
            rng = np.random.default_rng(seed)
            samples = X_minority[rng.integers(0, n_min, size=n_synthetic)]
            source = "bootstrap"
    out = np.asarray(samples.astype(np.float32))
    return (out, source) if return_source else out
