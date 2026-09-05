"""Latent factors and the observed columns that reflect them: multicollinearity with a known cause.

Correlated columns can be produced two ways, and the difference decides what a benchmark on them measures.
Drawing from a multivariate normal with a chosen covariance gives correlation with no structure behind it.
Drawing a latent factor and emitting reflections of it gives correlation WITH a cause, which is what makes
the interesting questions expressible: whether an arm collapses a redundant cluster, whether the collapse
costs anything, and whether the latent or its proxies belong in the answer key.

``distinct_sd`` is the reason this module is not just a covariance draw. With ``distinct_sd = 0`` the
reflections are noisy copies and averaging them is lossless -- collapsing the cluster is correct. With
``distinct_sd > 0`` each reflection carries private information, and the target is driven by an UNEQUALLY
weighted combination of those private parts, so the cluster mean no longer contains what drives the target
and a redundancy-collapsing selector destroys signal it cannot recover.

The unequal weighting is the load-bearing detail and it was learned the hard way. An earlier version of this
scenario summed the private deltas with equal weights, which preserves exactly the combination the mean
carries -- so aggregation cost nothing measurable (0.0001 AUC) while the scenario's own docstring claimed it
was destructive. Weights must not be a permutation-invariant function of the members, or the mean is a
sufficient statistic for the part that matters and the scenario proves nothing.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from mlframe.data.datasets._columns import standardize
from mlframe.data.datasets._rng import stream_for
from mlframe.data.datasets.ground_truth import RedundancyGroup
from mlframe.data.datasets.spec import LatentSpec, resolve_knob

logger = logging.getLogger(__name__)

__all__ = ["LatentRealization", "delta_weights", "realize_latent", "realize_latents"]


class LatentRealization:
    """One realised latent factor: the factor itself, its reflections, and their private parts."""

    def __init__(self, name: str, values: np.ndarray, reflections: Dict[str, np.ndarray], deltas: Dict[str, np.ndarray], scales: Dict[str, float]) -> None:
        """Store the realised arrays.

        Args:
            name: The latent's name.
            values: The latent factor, one value per row.
            reflections: Observed columns generated from the latent.
            deltas: Each reflection's private deviation, empty when ``distinct_sd`` was zero.
            scales: Pre-standardisation scale per reflection.
        """
        self.name = name
        self.values = values
        self.reflections = reflections
        self.deltas = deltas
        self.scales = scales

    def redundancy_group(self) -> RedundancyGroup:
        """Return the redundancy record for this latent's reflections.

        The group is EXACT only when no reflection carries a private part: with private parts the members
        are not interchangeable, and declaring them so would tell a scoring layer that collapsing them is
        free when the scenario exists to prove it is not.
        """
        exact = not self.deltas
        return RedundancyGroup(
            members=tuple(sorted(self.reflections)),
            rank=1 if exact else 1 + len(self.deltas),
            exact=exact,
            source=self.name,
        )


def delta_weights(count: int) -> Tuple[float, ...]:
    """Return weights for the private parts that no permutation-invariant summary can reproduce.

    Alternating signs with geometrically decaying magnitudes. Any equally weighted scheme -- the obvious
    one -- makes the cluster MEAN a sufficient statistic for the driving combination, so averaging the
    cluster preserves the signal and the scenario silently stops testing what it claims to test.
    """
    return tuple(float(((-1.0) ** i) * (1.0 / (1.6**i))) for i in range(count))


def realize_latent(
    latent: LatentSpec,
    n: int,
    root_seed: int,
    spec_name: str,
    knob_rng: Optional[np.random.Generator] = None,
) -> LatentRealization:
    """Draw one latent factor and the observed reflections it generates.

    Each reflection is ``loading * z + distinct_sd * delta + noise_sd * e``. The private ``delta`` is drawn
    per reflection and kept, because the target's link consumes it directly -- that is what makes the
    reflections jointly necessary rather than interchangeable.

    Args:
        latent: The latent's declaration.
        n: Number of rows.
        root_seed: The dataset's root seed.
        spec_name: Dataset name, which namespaces every stream.
        knob_rng: Stream used to resolve prior-valued knobs; defaults to the latent's own stream.

    Returns:
        A :class:`LatentRealization`.
    """
    rng = stream_for(root_seed, spec_name, "latent", latent.name)
    resolver = knob_rng if knob_rng is not None else rng
    distinct_sd = resolve_knob(latent.distinct_sd, resolver)
    noise_sd = resolve_knob(latent.noise_sd, resolver)

    values = rng.normal(0.0, 1.0, n) if latent.family == "normal" else rng.standard_t(4.0, n)
    loadings = latent.loadings or tuple(1.0 for _ in latent.reflections)

    reflections: Dict[str, np.ndarray] = {}
    deltas: Dict[str, np.ndarray] = {}
    scales: Dict[str, float] = {}
    for loading, name in zip(loadings, latent.reflections):
        column = float(loading) * values
        if distinct_sd > 0.0:
            delta = stream_for(root_seed, spec_name, "latent", latent.name, "delta", name).normal(0.0, 1.0, n)
            deltas[name] = delta
            column = column + distinct_sd * delta
        if noise_sd > 0.0:
            column = column + noise_sd * stream_for(root_seed, spec_name, "latent", latent.name, "noise", name).normal(0.0, 1.0, n)
        scaled, scale = standardize(column)
        reflections[name] = scaled
        scales[name] = scale

    return LatentRealization(name=latent.name, values=values, reflections=reflections, deltas=deltas, scales=scales)


def realize_latents(
    latents: Tuple[LatentSpec, ...],
    n: int,
    root_seed: int,
    spec_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], List[RedundancyGroup]]:
    """Realise every latent, returning its columns, the latent factors, the scales and the redundancy groups.

    Reflections overwrite any column of the same name drawn by the marginal layer: a reflection is defined
    by its latent, and a spec that declares both is telling the generator the column is a reflection.
    """
    columns: Dict[str, np.ndarray] = {}
    factors: Dict[str, np.ndarray] = {}
    scales: Dict[str, float] = {}
    groups: List[RedundancyGroup] = []
    for latent in latents:
        realization = realize_latent(latent, n, root_seed, spec_name)
        columns.update(realization.reflections)
        factors[latent.name] = realization.values
        scales.update(realization.scales)
        if len(realization.reflections) > 1:
            groups.append(realization.redundancy_group())
        if realization.deltas:
            # The link layer needs the private parts to build a combination the cluster mean cannot carry.
            factors.update({f"{latent.name}::delta::{name}": delta for name, delta in realization.deltas.items()})
    return columns, factors, scales, groups
