"""Synthetic-noise-then-recentering trend decorrelation for Kaggle-style segments.

COMPETITION / EXPLORATORY ONLY — NEVER wire this into production defaults.

Source: 1st_lanl-earthquake-prediction.md — the winning solution found, via
adversarial validation (training a classifier to distinguish train vs. test
segments), that a spurious non-stationary time-trend artifact in the raw
150k-sample acoustic segments was separable by that classifier. Rather than
addressing why train and test differ, they defeated the adversarial detector
by injecting fixed-seed Gaussian noise into each segment and then subtracting
the segment's median, which erases the low-frequency trend component that the
adversarial classifier was keying on while leaving peak/volatility structure
(the real predictive signal for the competition's target) largely intact.

This is DELIBERATE signal destruction, targeted narrowly at defeating one
specific adversarial-validation classifier on one specific competition's
segment structure. It is an actively harmful practice in a real production
model: production feature engineering must never inject noise to hide a
real (or spurious) distributional shift from a diagnostic — a shift flagged
by adversarial validation in production is a signal to investigate the data
pipeline, not to launder away with noise. Any production use of this pattern
would silently degrade the honesty of downstream monitoring and could mask
genuine train/serving skew. It lives under ``mlframe.competition`` and must
never be imported by production mlframe modules or exported from mlframe's
top-level ``__init__.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = ["inject_noise_and_recenter"]


def inject_noise_and_recenter(
    segment: npt.ArrayLike,
    noise_std: float = 0.5,
    random_state: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """Inject fixed-seed Gaussian noise into `segment`, then subtract its (post-noise) median.

    COMPETITION / EXPLORATORY ONLY — see module docstring. Deliberately
    destroys low-frequency trend structure to defeat an adversarial-validation
    classifier trained to detect train/test distributional shift; never use
    this to hide a real shift in a production pipeline.

    Parameters
    ----------
    segment:
        1-D array of raw time-segment values.
    noise_std:
        Standard deviation of the injected zero-mean Gaussian noise.
    random_state:
        Seed for the noise draw. Fixed seeding is intentional (reproducible
        adversarial-defeat), not a general-purpose RNG convenience.

    Returns
    -------
    The noised-and-recentered segment as ``float64``.
    """
    x = np.asarray(segment, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("segment must be 1-D")
    if x.size == 0:
        raise ValueError("segment must be non-empty")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")

    rng = np.random.default_rng(random_state)
    noised = x + rng.normal(0.0, noise_std, size=x.shape)
    return np.asarray(noised - np.median(noised), dtype=np.float64)
