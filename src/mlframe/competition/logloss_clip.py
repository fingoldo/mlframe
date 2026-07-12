"""Probability clipping before log-loss scoring (competition leaderboard trick).

*** COMPETITION / EXPLORATORY ONLY - NOT PRODUCTION CODE ***

This module implements a Kaggle-competition-specific trick (Tabular Playground
Series May 2021 1st place solution: clipping predicted probabilities to
``[0.05, 0.95]`` before computing log-loss) to bound the penalty a confidently-wrong
prediction inflicts on the log-loss metric.

Why this is NOT production-safe:
    - This is metric-gaming, not model improvement. Clipping does not change a
      single prediction's rank or the model's actual calibration - it only
      truncates the *loss function's* worst-case penalty, artificially inflating
      the reported log-loss score without making the model any better (and,
      per the tracker critique, "actively degrades calibration honesty by
      artificially bounding confidence").
    - A proper scoring rule (log-loss, Brier score) is proper precisely because
      it rewards honest, well-calibrated probabilities including confident ones
      near 0/1. Clipping breaks propriety: it can make a systematically
      overconfident-and-wrong model look better on log-loss while doing nothing
      (or actively hurting) other honest metrics like Brier score or ECE.
    - Never apply this before scoring/selecting a production model, and never
      apply it to the probabilities actually shipped to users/downstream
      systems - it silently distorts probabilities that may be consumed as
      real risk estimates.

This module lives under ``mlframe.competition`` and is NEVER imported by any
production mlframe module, and NEVER re-exported from mlframe's top-level
``__init__.py``. Use it only for exploratory leaderboard-hunting work, and only
to score against a log-loss-based competition metric - never to "improve" a
model or to post-process production-facing probabilities.

Default bounds here are deliberately tight (``[1e-4, 1 - 1e-4]``) - just enough
to avoid literal +inf/-inf log-loss from a 0.0/1.0 prediction. The aggressive
``[0.05, 0.95]`` bound from the source competition is supported via explicit
arguments for anyone reproducing that specific leaderboard trick, but is NOT
the default precisely because it is the metric-gaming regime the critique
warns about.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LOWER = 1e-4
DEFAULT_UPPER = 1 - 1e-4


def clip_probabilities_for_logloss(
    probs: np.ndarray,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> np.ndarray:
    """Clip predicted probabilities into ``[lower, upper]`` before log-loss scoring.

    *** COMPETITION / EXPLORATORY ONLY - see module docstring. This is a metric-
    gaming trick that bounds the log-loss penalty of confidently-wrong
    predictions without improving the underlying model; it can make an honest
    proper-scoring-rule metric (Brier score, calibration error) look WORSE
    while making log-loss look better. Never use in production scoring or on
    probabilities served to downstream consumers. ***

    Args:
        probs: array of predicted probabilities (any shape), expected in [0, 1].
        lower: lower clip bound. Defaults to a numerically-safe ``1e-4`` (just
            enough to avoid infinite log-loss on a literal 0.0 prediction).
            Pass ``0.05`` to reproduce the aggressive competition-leaderboard
            trick from the source solution.
        upper: upper clip bound. Defaults to ``1 - 1e-4``. Pass ``0.95`` to
            reproduce the aggressive competition trick.

    Returns:
        A new array of the same shape as ``probs``, clipped into ``[lower, upper]``.

    Raises:
        ValueError: if ``lower`` is not in ``[0, 1)``, ``upper`` is not in
            ``(0, 1]``, or ``lower >= upper``.
    """
    if not (0.0 <= lower < 1.0):
        raise ValueError(f"lower must be in [0, 1), got {lower}")
    if not (0.0 < upper <= 1.0):
        raise ValueError(f"upper must be in (0, 1], got {upper}")
    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be < upper ({upper})")

    probs_arr = np.asarray(probs, dtype=np.float64)
    return np.clip(probs_arr, lower, upper)
