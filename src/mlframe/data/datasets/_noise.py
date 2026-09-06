"""Label corruptions, each of which must state how it moves ``true_prob``.

This module enforces the invariant the whole ceiling story rests on: **a corruption that cannot supply an
update to ``true_prob`` is refused.** The alternative -- corrupt the labels and leave the recorded
probabilities describing the clean data -- produces a ceiling that is quietly wrong, and every regret
figure measured against it is wrong by the same unknown amount with nothing to flag it.

The updates are all elementary, which is what makes the rule affordable rather than aspirational:

* a uniform flip at rate ``f`` sends ``p`` to ``p (1 - f) + (1 - p) f``;
* a feature-dependent flip is the same expression with ``f`` varying by row, so the corrupted target's
  Bayes rule now depends on the gating column -- which is a real, and interesting, structural change rather
  than added noise;
* quantising the probability into bins replaces each row's ``p`` with its bin's mean, which is exactly what
  a downstream model with limited resolution sees.

An earlier design proposed refusing corruptions without a closed-form ceiling instead. That rule would have
deleted the two families where the ceiling matters most -- feature-dependent flips and missingness -- while
letting through cases whose "closed form" was itself a Monte-Carlo estimate over realised probabilities.
Requiring the update, and taking the ceiling by exact computation over the updated probabilities, keeps both
the families and the guarantee.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import numpy as np

from mlframe.data.datasets._links import gate_mask
from mlframe.data.datasets.spec import NoiseSpec, resolve_knob

logger = logging.getLogger(__name__)

__all__ = ["uniform_flip", "feature_dependent_flip", "binning_pushforward", "apply_corruption"]


def uniform_flip(probability: np.ndarray, rate: float) -> np.ndarray:
    """Return the post-flip probability for a label flipped with a constant rate."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"flip rate must lie in [0, 1]; got {rate}")
    return probability * (1.0 - rate) + (1.0 - probability) * rate


def feature_dependent_flip(probability: np.ndarray, rate_per_row: np.ndarray) -> np.ndarray:
    """Return the post-flip probability when the flip rate varies by row.

    The corrupted target's Bayes-optimal predictor now depends on whatever drives ``rate_per_row``, so a
    column that gates the noise is informative about the observed label even if it never touched the clean
    one. That is a structural change worth testing, not merely a louder version of uniform noise.
    """
    if np.any(rate_per_row < 0.0) or np.any(rate_per_row > 1.0):
        raise ValueError("per-row flip rates must lie in [0, 1]")
    flipped: np.ndarray = probability * (1.0 - rate_per_row) + (1.0 - probability) * rate_per_row
    return flipped


def binning_pushforward(probability: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the probability a model with ``n_bins`` of resolution sees: each row's bin mean.

    Equal-width bins on the probability scale, so the transformation is a function of ``p`` alone and does
    not depend on the sample -- an equal-mass version would make the corruption depend on which rows were
    drawn, and two datasets from the same spec would then carry different corruptions.
    """
    if n_bins < 2:
        raise ValueError(f"binning needs at least two bins; got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return np.asarray(centres[index], dtype=np.float64)


def apply_corruption(
    probability: np.ndarray,
    noise: NoiseSpec,
    columns: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    knob_rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """Apply one declared corruption to ``true_prob`` and return the updated probabilities.

    Labels are not drawn here. The corruption is expressed entirely as a transformation of the probability,
    and the label is sampled once from the final probability -- which is both simpler and strictly correct,
    since sampling then flipping and sampling from the flipped probability have the same law.

    Args:
        probability: Clean per-row probabilities.
        noise: The corruption declaration.
        columns: Realised columns, needed by a gated corruption.
        rng: Stream for any randomness the corruption itself needs.
        knob_rng: Stream for resolving a prior-valued rate.

    Returns:
        ``(updated_probability, caveat)``, where the caveat is a sentence for the truth record's caveat list
        or ``None`` when the corruption leaves nothing to warn about.

    Raises:
        ValueError: If the declared corruption has no matching ``true_prob`` update. The spec layer already
            refuses this combination, so reaching it here means the two layers disagree -- which is worth a
            loud failure rather than a silent pass.
    """
    if noise.kind == "none":
        return probability, None

    rate = resolve_knob(noise.rate, knob_rng or rng)

    if noise.kind == "uniform_flip":
        return uniform_flip(probability, float(rate)), None

    if noise.kind == "feature_dependent_flip":
        if noise.gate is None:
            raise ValueError("feature_dependent_flip reached the generator without a gate")
        inside = gate_mask(noise.gate, columns)
        rates = np.where(inside, float(rate), 0.0)
        caveat = (
            f"labels are flipped at rate {float(rate):.3g} only where {noise.gate.column!r} is inside its gate, "
            "so that column is informative about the OBSERVED target even where it does not drive the clean one"
        )
        return feature_dependent_flip(probability, rates), caveat

    if noise.kind == "binning":
        n_bins = max(2, round(1.0 / float(rate))) if rate else 10
        caveat = f"probabilities are quantised to {n_bins} equal-width bins, so the ceiling is the quantised one"
        return binning_pushforward(probability, n_bins), caveat

    raise ValueError(f"corruption kind {noise.kind!r} has no true_prob update; refusing to corrupt the ceiling")
