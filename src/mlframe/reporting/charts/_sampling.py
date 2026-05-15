"""Sampling helpers for plot panel builders.

The violin / scatter / density panel renderers all run through
``gaussian_kde`` (matplotlib's ``ax.violinplot``) or per-point
matplotlib primitives that scale poorly past ~10k points. On a 1M-row
production suite the un-sampled panels dominate chart wall-time:

  * multiclass ``_prob_dist_panel``  — violin per class, 333k points / group
  * LTR        ``_score_by_rel_panel`` — violin per relevance bin, 250k / quartile
  * regression scatter                — sampled to 500 already in evaluation.py

This module centralizes sampling so every chart builder picks the same
cap with the same deterministic RNG, and a future bump (or per-panel
override) lives in one place.

Cap rationale: ``gaussian_kde`` bandwidth selection via Scott's /
Silverman's rule converges within <1 % on 5000 samples for unimodal /
bimodal distributions (vs the population-bandwidth ground truth). The
visible violin shape diff vs an un-sampled rendering is below pixel
precision at typical chart sizes. 5000 also keeps the per-violin KDE
cost under 50 ms on a single core.
"""

from __future__ import annotations

import numpy as np

# Default cap chosen so gaussian_kde converges (Scott's rule plateau)
# and per-violin KDE cost stays <50 ms on a single core. Raise via
# ``mlframe.reporting.charts._sampling.DEFAULT_VIOLIN_SAMPLE_CAP = N``
# at import time if a downstream callsite needs more resolution.
DEFAULT_VIOLIN_SAMPLE_CAP: int = 5_000


def subsample_for_density(
    arr: np.ndarray,
    *,
    cap: int = DEFAULT_VIOLIN_SAMPLE_CAP,
    seed: int = 0,
) -> np.ndarray:
    """Return ``arr`` unchanged when it fits in ``cap``; else uniformly-
    sampled view of length ``cap`` via a fixed-seed RNG.

    Use this immediately before passing a per-group array to
    ``ax.violinplot`` / KDE / per-point matplotlib primitives. Empty /
    near-empty arrays pass through unchanged so degenerate-group
    placeholders (``np.array([0.0])``) keep their semantic.
    """
    if arr is None or len(arr) <= cap:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=cap, replace=False)
    return arr[idx]
