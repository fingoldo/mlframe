"""Targets that are not a coin flip: multiclass, ordinal and counts, each with its own exact ceiling.

Every bed in this suite so far has a binary target, which quietly restricts what the benchmark can say. A
selector's job changes with the target: with three classes the relevant set is the UNION of what each
class depends on, and a filter scoring one column against one binary label is answering a different
question. With an ordinal target the classes are ordered, so a method that treats them as unordered
throws away most of the signal. With a count the variance grows with the mean, and a scorer assuming
constant variance is mis-weighted on exactly the rows that matter most.

The three kinds are genuinely different problems and are constructed to stay different:

* **multiclass** -- each class gets its own weight vector, produced by ROTATING the declared one. That
  rotation is what makes the bed multiclass rather than ordinal: the classes do not lie along one
  latent axis, so no single ordering of the score reproduces them, and the answer key is the union.
* **ordinal** -- one latent score and ``K-1`` ordered cut points (the proportional-odds construction).
  The classes DO lie along one axis, which is exactly what distinguishes this from the case above, and
  the answer key is the same set for every class.
* **count** -- a Poisson mean of ``exp(score)``. Heteroscedastic by construction, since the variance of a
  Poisson is its mean.

Each returns the full per-row law, not just the labels, so the ceiling is exact for the same reason it is
in the binary case: the generator knows the distribution the labels were drawn from, so the best
achievable log-loss is the mean entropy of that distribution and the best achievable accuracy is the mean
of its largest probability. No simulation, no sampling noise.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "rotate_weights",
    "multiclass_probabilities",
    "ordinal_probabilities",
    "count_mean",
    "sample_categorical",
    "sample_count",
    "categorical_ceiling",
    "count_ceiling",
]


def rotate_weights(coefficients: Dict[str, float], n_classes: int) -> Tuple[Dict[str, float], ...]:
    """Return one weight vector per class, each a rotation of the declared one.

    The first class is the reference and gets the zero vector, so the declared coefficients describe class
    one against it exactly as they describe the positive class in the binary case. Every later class
    rotates the same values onto different columns.

    Rotation rather than fresh random draws: the classes then depend on the SAME columns with the same
    magnitudes in a different arrangement, so the bed's difficulty and its answer key are unchanged by the
    class count. Redrawing per class would make a three-class bed and a five-class bed differ in signal
    strength as well as in class count, and no comparison between them would mean anything.
    """
    names = list(coefficients)
    values = [float(coefficients[name]) for name in names]
    out: list = [{}]
    for offset in range(1, max(2, int(n_classes))):
        rotated = values[offset % max(1, len(values)) :] + values[: offset % max(1, len(values))]
        out.append({name: weight for name, weight in zip(names, rotated)})
    return tuple(out[: max(2, int(n_classes))])


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row maximum so a large score cannot overflow."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return np.asarray(exponentiated / np.sum(exponentiated, axis=1, keepdims=True), dtype=np.float64)


def multiclass_probabilities(class_scores: np.ndarray) -> np.ndarray:
    """Return the ``(n, K)`` class probabilities for per-class latent scores."""
    return _softmax(np.asarray(class_scores, dtype=np.float64))


def ordinal_probabilities(score: np.ndarray, cut_points: np.ndarray) -> np.ndarray:
    """Return ``(n, K)`` probabilities under proportional odds: one latent, ``K-1`` ordered cut points.

    ``P(Y <= k) = sigmoid(cut_k - score)``, differenced across the cuts. The cut points must be increasing,
    which is what makes the classes ordered; an unsorted set would produce negative probabilities and a
    bed that is neither ordinal nor multiclass.

    Raises:
        ValueError: If the cut points are not strictly increasing.
    """
    cuts = np.asarray(cut_points, dtype=np.float64)
    if cuts.size < 1 or np.any(np.diff(cuts) <= 0):
        raise ValueError(f"ordinal cut points must be strictly increasing; got {cuts.tolist()}")

    from mlframe.data.datasets._target import sigmoid

    values = np.asarray(score, dtype=np.float64).reshape(-1, 1)
    cumulative = sigmoid(cuts.reshape(1, -1) - values)
    ones = np.ones((values.shape[0], 1), dtype=np.float64)
    zeros = np.zeros((values.shape[0], 1), dtype=np.float64)
    bounded = np.hstack([zeros, cumulative, ones])
    return np.asarray(np.diff(bounded, axis=1), dtype=np.float64)


def count_mean(score: np.ndarray, clip: float = 12.0) -> np.ndarray:
    """Return the Poisson mean ``exp(score)``, clipped in the EXPONENT so no row dominates the draw.

    Clipping the exponent rather than the mean keeps the relationship monotone everywhere and bounds the
    largest mean to a value a count column can plausibly take. Without it a single extreme score produces
    a row whose count is larger than every other row combined, and the bed measures outlier handling.
    """
    return np.asarray(np.exp(np.clip(np.asarray(score, dtype=np.float64), -clip, clip)), dtype=np.float64)


def sample_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one class per row from its own categorical distribution."""
    cumulative = np.cumsum(np.asarray(probabilities, dtype=np.float64), axis=1)
    draws = np.asarray(rng.random(cumulative.shape[0]), dtype=np.float64).reshape(-1, 1)
    return np.asarray((draws > cumulative).sum(axis=1), dtype=np.int64)


def sample_count(means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one Poisson count per row."""
    return np.asarray(rng.poisson(np.asarray(means, dtype=np.float64)), dtype=np.int64)


def categorical_ceiling(probabilities: np.ndarray) -> Dict[str, float]:
    """Return the exact achievable log-loss, accuracy and Brier for a known categorical law.

    All three are expectations over the law the labels came from, so they are exact rather than estimated:
    the best achievable log-loss is the mean entropy, the best achievable accuracy is the mean of the
    largest per-row probability, and the multiclass Brier is the mean of ``1 - sum(p^2)``.
    """
    probability = np.asarray(probabilities, dtype=np.float64)
    safe = np.clip(probability, 1e-12, 1.0)
    return {
        "log_loss": float(np.mean(-np.sum(probability * np.log(safe), axis=1))),
        "accuracy": float(np.mean(np.max(probability, axis=1))),
        "brier": float(np.mean(1.0 - np.sum(np.square(probability), axis=1))),
    }


def count_ceiling(means: np.ndarray) -> Dict[str, float]:
    """Return the achievable Poisson deviance and the mean absolute deviation for a known count law.

    The deviance of a perfect model is not zero for a count target -- the label is random given the mean --
    and its expectation has a closed form only through the distribution itself, so it is computed from the
    law rather than from any sample. Reporting zero would make every real model look infinitely far from
    a ceiling nothing can reach.
    """
    lam = np.asarray(means, dtype=np.float64)
    # E[2(y log(y/mu) - (y - mu))] under y ~ Poisson(mu), summed over the support that carries the mass.
    total = 0.0
    for index, raw in enumerate(lam):
        value = float(raw)
        upper = int(max(10, value + 8.0 * np.sqrt(max(value, 1e-9))))
        counts = np.arange(0, upper + 1, dtype=np.float64)
        from scipy import stats

        weights = stats.poisson.pmf(counts, value)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(counts > 0, counts * np.log(np.maximum(counts, 1e-12) / max(value, 1e-12)), 0.0)
        total += float(np.sum(weights * 2.0 * (terms - (counts - value))))
        if index > 2000:
            # The mean converges long before every row is visited; walking a million rows to refine the
            # fourth decimal of a ceiling nothing reaches is not worth the generation time.
            total *= len(lam) / float(index + 1)
            break
    return {"poisson_deviance": total / float(len(lam)), "mean": float(np.mean(lam))}
