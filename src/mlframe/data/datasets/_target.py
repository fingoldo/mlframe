"""Score to probability to label, plus the calibration that makes difficulty comparable across links.

Two things happen here that decide whether a scenario measures what it claims.

**Prevalence is set by moving the intercept, not by resampling rows.** Dropping majority rows to reach a 1%
positive rate changes the number of rows AND the class balance at once, so a method that degrades with
sample size looks like a method that degrades with imbalance. Shifting the intercept until the mean
probability hits the target leaves every row and every feature untouched.

**Difficulty is set by calibrating to an achievable ceiling, not by choosing coefficients.** A coefficient
of 0.8 means something entirely different under a logistic link than under a parity gate, so a suite that
pins coefficients is comparing links rather than methods. The link scale is bisected until the Bayes AUC
hits a declared target, which is what makes a recovery-versus-ceiling CURVE possible -- and the point where
two methods cross on that curve is a finding, whereas their ranking at one hand-picked signal level is a
choice made by whoever picked it.

The Bayes AUC itself is computed exactly from the realised probabilities rather than simulated. For the
Bayes-optimal score the ranking IS the probability, so the AUC is the probability that a random positive
outranks a random negative, and both are known per row: the numerator is a sum over ordered pairs weighted
by ``p_i (1 - p_j)``, which a sort turns into a linear pass. No sampling noise, no seed dependence, no
convergence question.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "sigmoid",
    "probability_from_score",
    "shift_to_prevalence",
    "bayes_auc",
    "bayes_brier",
    "calibrate_scale",
    "sample_labels",
]

_BISECTION_STEPS = 60


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_v = np.exp(values[~positive])
    out[~positive] = exp_v / (1.0 + exp_v)
    return out


def probability_from_score(score: np.ndarray, kind: str = "logistic") -> np.ndarray:
    """Map a latent score to a probability.

    ``threshold`` is deliberately not fully deterministic: probabilities of exactly zero and one make the
    sample log-loss ceiling zero and attainable only in the limit, and a scenario that wants that must ask
    for it explicitly rather than getting it as a side effect of choosing a link.
    """
    if kind == "threshold":
        return np.asarray(np.clip(0.5 + 0.45 * np.sign(score), 0.01, 0.99), dtype=np.float64)
    return sigmoid(score)


def shift_to_prevalence(score: np.ndarray, target: float, kind: str = "logistic") -> Tuple[np.ndarray, float]:
    """Return ``(probabilities, intercept_shift)`` whose mean matches ``target``.

    Bisection on the intercept rather than row resampling: the alternative changes sample size and balance
    together, which confounds imbalance with sample size in every downstream comparison.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"prevalence must lie strictly inside (0, 1); got {target}")
    low, high = -30.0, 30.0
    shift = 0.0
    for _ in range(_BISECTION_STEPS):
        shift = 0.5 * (low + high)
        mean = float(np.mean(probability_from_score(score + shift, kind)))
        if mean < target:
            low = shift
        else:
            high = shift
    return probability_from_score(score + shift, kind), shift


def bayes_auc(probability: np.ndarray) -> float:
    """Return the exact AUC of the Bayes-optimal score on the realised rows.

    For the optimal score the ranking is the probability itself, so

        AUC = sum_{i,j} p_i (1 - p_j) [p_i > p_j] + 0.5 * [p_i == p_j], normalised by sum_{i,j} p_i (1 - p_j)

    Sorting once turns the double sum into cumulative sums, so this is exact and O(n log n) rather than
    estimated by simulation -- which matters because the ceiling is what every regret figure is measured
    against, and a noisy denominator would propagate into all of them.
    """
    p = np.asarray(probability, dtype=np.float64).ravel()
    if p.size < 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    q_sorted = 1.0 - p_sorted
    cumulative_q = np.concatenate(([0.0], np.cumsum(q_sorted)))

    # Strictly-smaller mass is measured to the start of a row's OWN tie group, not to its own position: a
    # prefix sum would count earlier members of the same group as smaller, which turns a bed where every
    # probability is identical -- every pair a tie, ceiling one half -- into a reported ceiling of one.
    less_mass = np.empty_like(p_sorted)
    tie_mass = np.empty_like(p_sorted)
    start = 0
    for end in range(1, p_sorted.size + 1):
        if end == p_sorted.size or p_sorted[end] != p_sorted[start]:
            group_q = cumulative_q[end] - cumulative_q[start]
            less_mass[start:end] = cumulative_q[start]
            tie_mass[start:end] = 0.5 * (group_q - q_sorted[start:end])
            start = end

    numerator = float(np.sum(p_sorted * (less_mass + tie_mass)))
    denominator = float(np.sum(p_sorted) * np.sum(q_sorted) - np.sum(p_sorted * q_sorted))
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def bayes_brier(probability: np.ndarray) -> float:
    """Return the Brier score of the Bayes-optimal predictor: the mean of ``p (1 - p)``.

    Exact in expectation over the labels given the realised probabilities, which is the honest comparator
    for an arm scored on those same rows.
    """
    p = np.asarray(probability, dtype=np.float64)
    return float(np.mean(p * (1.0 - p)))


def calibrate_scale(
    probability_at: Callable[[float], np.ndarray],
    target_auc: float,
    low: float = 1e-3,
    high: float = 50.0,
) -> Tuple[float, float]:
    """Bisect the link scale until the Bayes AUC of the FINAL probabilities reaches ``target_auc``.

    The caller supplies ``probability_at(scale)``, which must apply everything that stands between the link
    score and the law the labels are drawn from -- the prevalence shift and any label corruption included.
    Calibrating on the clean pre-corruption probabilities instead is the tempting shortcut and it is wrong:
    a scenario declaring "AUC 0.80 achievable" and then flipping 5% of its labels ships data whose real
    ceiling is 0.76, and every regret measured against the declared number is off by that gap.

    Args:
        probability_at: Maps a candidate link scale to the final per-row probabilities.
        target_auc: Achievable AUC the scenario declares as its difficulty.
        low: Lower bracket for the scale.
        high: Upper bracket.

    Returns:
        ``(scale, achieved_auc)``. When the target is unreachable inside the bracket the closest endpoint is
        returned with the AUC it achieves, so the caller can record the miss rather than assume a hit.
    """
    if not 0.5 < target_auc < 1.0:
        raise ValueError(f"target AUC must lie strictly inside (0.5, 1.0); got {target_auc}")

    def achieved(scale: float) -> float:
        """Return the Bayes AUC of the final probabilities at one candidate scale."""
        return bayes_auc(probability_at(scale))

    lo_auc, hi_auc = achieved(low), achieved(high)
    if not np.isfinite(lo_auc) or not np.isfinite(hi_auc):
        logger.warning("ceiling calibration got a non-finite AUC at a bracket end; returning the unit scale")
        return 1.0, achieved(1.0)
    if target_auc <= lo_auc:
        return low, lo_auc
    if target_auc >= hi_auc:
        # The declared difficulty is beyond what this structure can reach: report the miss instead of
        # silently returning the widest scale as though it had been achieved.
        logger.warning("target AUC %.3f exceeds the reachable %.3f at scale %.3g", target_auc, hi_auc, high)
        return high, hi_auc

    lo, hi = low, high
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if achieved(mid) < target_auc:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    return scale, achieved(scale)


def sample_labels(probability: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw binary labels from per-row probabilities."""
    return (rng.random(probability.shape[0]) < probability).astype(np.int64)
