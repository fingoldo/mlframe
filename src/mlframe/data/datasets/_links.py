"""From columns to the latent score that drives the target: additive terms, interactions, gates.

The link is where a scenario states what kind of dependence it is testing, and the kinds are genuinely
different problems rather than degrees of one:

* ``linear`` and ``logistic`` -- every operand is visible marginally, which is the easy case every method
  handles and the one where a benchmark learns nothing.
* ``parity`` -- the operands have ZERO marginal mutual information with the target by construction and only
  become informative jointly, so any marginally greedy ranking is blind to them. This is the case that
  separates methods, and it is the case a multiplicative "interaction" does NOT produce: ``x * y`` leaves
  both operands marginally informative, which is how two builders in this repository ended up named after
  parity while testing nothing of the sort.
* ``threshold`` and the regional gate -- an effect that exists on part of the input space and nowhere else.
  A global summary statistic averages it away, which is a failure mode invisible in every global metric.

Heteroscedasticity lives here rather than in the noise layer on purpose. Making the noise scale depend on a
feature changes how much of the target that feature explains WHERE, so it belongs with the structure that
decides the score, not with the corruption applied to labels afterwards. The features driving the variance
are recorded, because a column that only affects the variance is informative in a way that a mean-based
importance measure cannot see -- a real failure mode, and one this generator can state as truth.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from mlframe.data.datasets.spec import GateSpec, LinkSpec, Prior, resolve_knob

logger = logging.getLogger(__name__)

__all__ = ["gate_mask", "parity_term", "tail_gate_term", "additive_score", "interaction_score", "link_score", "apply_heteroscedasticity"]


def gate_mask(gate: GateSpec, columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the boolean mask of rows inside a gate's region.

    A gate is stated either as an explicit interval or as a fraction of the column's own distribution; the
    fraction form is what keeps a region the same SIZE across marginal families, so a regional effect on a
    lognormal column covers the same share of rows as on a normal one.

    Args:
        gate: The region declaration.
        columns: Realised columns, which must contain ``gate.column``.

    Returns:
        A boolean array, one entry per row.

    Raises:
        KeyError: If the gate names a column the dataset does not have.
    """
    if gate.column not in columns:
        raise KeyError(f"gate references column {gate.column!r}, which the dataset does not contain")
    values = columns[gate.column]
    if gate.fraction is not None:
        threshold = float(np.quantile(values, 1.0 - float(gate.fraction)))
        return values >= threshold
    low = -np.inf if gate.low is None else float(gate.low)
    high = np.inf if gate.high is None else float(gate.high)
    return (values >= low) & (values <= high)


def parity_term(operands: Sequence[np.ndarray]) -> np.ndarray:
    """Return the parity of the operands' signs, as ``-1``/``+1``.

    Every operand is marginally independent of the result when the operands are symmetric around zero: the
    parity flips with equal probability whichever side of zero a single operand falls on. That is the whole
    point, and it is what a product term fails to deliver.
    """
    bits = np.ones(operands[0].shape[0], dtype=np.float64)
    for values in operands:
        bits = bits * np.where(values > 0.0, 1.0, -1.0)
    return bits


def additive_score(coefficients: Mapping[str, float], columns: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    """Return the weighted sum of the named columns.

    Raises:
        KeyError: If a coefficient names a column that does not exist, which is a scenario typo that would
            otherwise silently drop a term and change what the scenario tests.
    """
    score = np.zeros(n, dtype=np.float64)
    for name, weight in coefficients.items():
        if name not in columns:
            raise KeyError(f"link coefficient references unknown column {name!r}")
        score = score + float(weight) * columns[name]
    return score


def tail_gate_term(operands: Sequence[np.ndarray], quantile: float) -> np.ndarray:
    """Return 1 where EVERY operand exceeds its own quantile, else 0.

    The signal then lives in the joint upper tail and nowhere else, which is the case a Gaussian copula
    cannot produce and an equal-mass binned estimator cannot see: at ten bins the whole joint tail is one
    cell of the joint histogram.
    """
    fires = np.ones(operands[0].shape[0], dtype=bool)
    for values in operands:
        fires &= values >= float(np.quantile(values, quantile))
    return fires.astype(np.float64)


def interaction_score(
    kind: str,
    interactions: Sequence[Sequence[str]],
    weights: Sequence[float],
    columns: Mapping[str, np.ndarray],
    n: int,
    tail_quantile: float = 0.9,
) -> np.ndarray:
    """Return the contribution of the interaction terms under one link kind.

    Under ``parity`` the term is the sign parity of its operands; under ``tail_gate`` it fires only where
    every operand is in its own upper tail; under every other kind it is their product. The distinctions are
    deliberate and are the difference between a bed that tests synergy blindness, one that tests tail
    blindness, and one that only looks like it does either.
    """
    score = np.zeros(n, dtype=np.float64)
    effective = list(weights) if weights else [1.0] * len(interactions)
    for term, weight in zip(interactions, effective):
        operands = []
        for name in term:
            if name not in columns:
                raise KeyError(f"interaction term references unknown column {name!r}")
            operands.append(columns[name])
        if kind == "parity":
            contribution = parity_term(operands)
        elif kind == "tail_gate":
            contribution = tail_gate_term(operands, tail_quantile)
        else:
            contribution = np.prod(np.vstack(operands), axis=0)
        score = score + float(weight) * contribution
    return score


def link_score(
    link: LinkSpec,
    columns: Mapping[str, np.ndarray],
    n: int,
    knob_rng: Optional[np.random.Generator] = None,
    scale_override: Optional[float] = None,
) -> np.ndarray:
    """Return the latent score, before any target-specific transformation.

    ``scale_override`` exists for the ceiling calibration, which bisects on the scale: the calibrator needs
    to evaluate the same score at many scales without rebuilding the spec, and rebuilding it would redraw
    the data and make the bisection non-monotone for a reason unrelated to difficulty.

    Args:
        link: The link declaration.
        columns: Realised columns.
        n: Number of rows.
        knob_rng: Stream for resolving a prior-valued scale.
        scale_override: Scale to use instead of the declared one.

    Returns:
        The score array.
    """
    score = additive_score(link.coefficients, columns, n)
    if link.interactions:
        score = score + interaction_score(link.kind, link.interactions, link.interaction_weights, columns, n, tail_quantile=link.tail_quantile)
    if link.kind == "polynomial":
        score = score + np.square(score) * 0.25
    if link.kind == "threshold":
        score = np.where(score > 0.0, 1.0, -1.0) * np.abs(score)

    if link.region is not None:
        # A regional effect is present inside its region and absent outside it, rather than merely weaker:
        # a scaled-down global effect is still visible to a global statistic, which is the failure mode
        # this construction exists to avoid.
        score = np.where(gate_mask(link.region, columns), score, 0.0)

    if scale_override is not None:
        return float(link.intercept) + float(scale_override) * score
    # A prior-valued scale needs a stream; an absent one is a caller error rather than something to
    # paper over with an arbitrary seed, because two runs would then differ for a reason no spec records.
    if isinstance(link.scale, Prior) and knob_rng is None:
        raise ValueError("link.scale is a Prior, so link_score needs a knob_rng to draw it from")
    scale = resolve_knob(link.scale, knob_rng if knob_rng is not None else np.random.default_rng(0))
    return float(link.intercept) + scale * score


def apply_heteroscedasticity(
    score: np.ndarray,
    variance_drivers: Mapping[str, float],
    columns: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    base_sd: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Add noise whose scale depends on the named columns, and report which columns drove it.

    The noise standard deviation is ``base_sd * exp(sum_j w_j * x_j / 2)``, i.e. log-linear in the drivers,
    which keeps it positive without a clip and makes the effect multiplicative in the natural way.

    A column appearing here and nowhere else in the link is informative about the target's UNCERTAINTY and
    not about its mean. Every mean-based importance measure -- correlation, a tree's split gain on the
    mean, a linear coefficient -- is blind to it, and calling it a noise column would be false. The drivers
    are returned so the truth record can say what they are.

    Args:
        score: The latent score to perturb.
        variance_drivers: ``{column: weight}`` controlling the log standard deviation.
        columns: Realised columns.
        rng: Noise stream.
        base_sd: Standard deviation when every driver sits at zero.

    Returns:
        ``(perturbed_score, drivers_used)``.
    """
    if not variance_drivers:
        return score, {}
    log_sd = np.zeros(score.shape[0], dtype=np.float64)
    used: Dict[str, float] = {}
    for name, weight in variance_drivers.items():
        if name not in columns:
            raise KeyError(f"variance driver references unknown column {name!r}")
        log_sd = log_sd + 0.5 * float(weight) * columns[name]
        used[name] = float(weight)
    sd = float(base_sd) * np.exp(np.clip(log_sd, -8.0, 8.0))
    return score + sd * rng.normal(0.0, 1.0, score.shape[0]), used
