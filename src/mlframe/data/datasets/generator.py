"""The generator: a pure function from a spec and a seed to data plus the truth that produced it.

``generate(spec)`` is deterministic in the spec alone. Every stream is addressed by name, so inserting a
column into the middle of a spec leaves every other column bit-identical -- the property that makes a
scenario library editable at all, and the one that positional ``SeedSequence.spawn`` quietly destroys.

Order of operations, and why it is this order:

1. exogenous marginals, so heavy tails, point masses and categorical levels exist before anything reads
   them;
2. latents and their reflections, which OVERWRITE columns of the same name -- a spec declaring both is
   saying the column is a reflection;
3. the link score at unit scale, then calibration to the declared ceiling, then prevalence;
4. corruption of ``true_prob``, never of the labels directly;
5. labels drawn once from the final probability.

Steps 3 to 5 are separated for a reason. Calibrating before setting prevalence and corrupting before drawing
labels keeps each stage a function of the previous stage's probabilities, so ``true_prob`` on the returned
truth is exactly the law the labels came from -- not an approximation of it, and not the clean law with a
corruption applied somewhere downstream where nothing records it.

The returned frame is pandas because that is what every consumer in this repository -- the arms, the
protocol layer, the downstream panel -- already takes. Categorical columns carry their declared levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mlframe.data.datasets._columns import draw_features
from mlframe.data.datasets._latent import realize_latents
from mlframe.data.datasets._links import apply_heteroscedasticity, link_score
from mlframe.data.datasets._noise import apply_corruption
from mlframe.data.datasets._rng import stream_for
from mlframe.data.datasets._scm import build_ground_truth
from mlframe.data.datasets._target import bayes_auc, bayes_brier, calibrate_scale, probability_from_score, sample_labels, shift_to_prevalence
from mlframe.data.datasets.ground_truth import GroundTruth
from mlframe.data.datasets.spec import DatasetSpec, TargetSpec, resolve_knob

logger = logging.getLogger(__name__)

__all__ = ["GeneratedDataset", "generate"]


@dataclass(frozen=True)
class GeneratedDataset:
    """A realised dataset: the frame, the labels, and the truth that generated them."""

    frame: pd.DataFrame
    target: pd.Series
    truth: GroundTruth
    calibration: Dict[str, Any]

    def as_tuple(self) -> Tuple[pd.DataFrame, pd.Series, GroundTruth]:
        """Return ``(frame, target, truth)`` for callers that prefer the positional form."""
        return self.frame, self.target, self.truth


def _assemble_frame(spec: DatasetSpec, columns: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Build the frame in the spec's declared column order, applying declared dtypes.

    The column order is never shuffled. ``sklearn.make_classification`` shuffles, which is precisely how it
    loses the ability to tell a caller which columns were informative; a generator whose entire purpose is
    to retain ground truth must not repeat that.
    """
    data: Dict[str, Any] = {}
    for feature in spec.features:
        values = columns[feature.name]
        if feature.dtype == "category":
            levels = feature.levels or ()
            codes = np.clip(values.astype(int), 0, max(len(levels) - 1, 0))
            data[feature.name] = pd.Categorical.from_codes(codes, categories=list(levels))
        elif feature.dtype == "int":
            data[feature.name] = np.rint(values).astype(np.int64)
        else:
            data[feature.name] = values.astype(np.float64)
    return pd.DataFrame(data, columns=list(spec.feature_names()))


def _variance_drivers(target: TargetSpec) -> Dict[str, float]:
    """Return the heteroscedasticity drivers a target declares, if any.

    Declared through the link's provenance-free extras rather than a dedicated field so that adding
    heteroscedasticity to a scenario does not change the hash of every spec that does not use it.
    """
    region = target.link.region
    if region is None or region.fraction is None:
        return {}
    # A region declared with a fraction doubles as a variance driver: the effect is present inside it and
    # the uncertainty differs outside, which is the shape most real regional effects actually have.
    return {region.column: 1.0}


def generate(spec: DatasetSpec, target_name: Optional[str] = None) -> GeneratedDataset:
    """Realise one dataset from its specification.

    Args:
        spec: The complete description, including its root seed.
        target_name: Which declared target to realise; defaults to the first.

    Returns:
        A :class:`GeneratedDataset` whose truth carries ``true_prob`` for the law the labels came from, the
        pre-standardisation scale of every column, and the calibration record.

    Raises:
        ValueError: If the spec declares no target, since a dataset without one has nothing to be truth
            about.
    """
    if not spec.targets:
        raise ValueError(f"spec {spec.name!r} declares no target")
    target = next((t for t in spec.targets if t.name == target_name), spec.targets[0])

    n = int(spec.n_samples)
    columns, scales = draw_features(spec.features, n, spec.root_seed, spec.name)
    latent_columns, factors, latent_scales, redundancy_groups = realize_latents(spec.latents, n, spec.root_seed, spec.name)
    columns.update(latent_columns)
    scales.update(latent_scales)

    # Private deltas are addressable by the link but never emitted as columns: they are what makes the
    # reflections jointly necessary, and exposing them would hand the answer to any selector.
    link_inputs: Dict[str, np.ndarray] = dict(columns)
    link_inputs.update(factors)

    knob_rng = stream_for(spec.root_seed, spec.name, "knobs", target.name)
    unit_score = link_score(target.link, link_inputs, n, knob_rng=knob_rng, scale_override=1.0)

    prevalence = resolve_knob(target.prevalence, knob_rng)
    corruption_rng = stream_for(spec.root_seed, spec.name, "corruption", target.name)
    drivers = _variance_drivers(target)
    used_drivers: Dict[str, float] = {}
    shift_seen = 0.0
    caveat_seen: Optional[str] = None

    def probability_at(candidate_scale: float) -> np.ndarray:
        """Return the final per-row probabilities a candidate link scale produces.

        Everything between the score and the law the labels come from lives inside here -- the
        heteroscedastic term, the prevalence shift and the corruption -- so the calibration bisects on the
        ceiling that actually ships rather than on a clean intermediate nobody ever observes. The
        heteroscedastic noise redraws per call from a fresh stream derived by name, so the bisection sees
        one consistent realisation rather than a moving target.
        """
        nonlocal used_drivers, shift_seen, caveat_seen
        candidate = candidate_scale * unit_score
        if drivers:
            candidate, used_drivers = apply_heteroscedasticity(candidate, drivers, columns, stream_for(spec.root_seed, spec.name, "hetero", target.name))
        shifted, shift_seen = shift_to_prevalence(candidate, float(prevalence), target.link.kind)
        corrupted, caveat_seen = apply_corruption(shifted, target.noise, columns, corruption_rng, knob_rng=knob_rng)
        return corrupted

    if target.calibrate_to is not None and target.calibrate_to.metric == "auc":
        scale, achieved = calibrate_scale(probability_at, float(target.calibrate_to.value))
        calibration: Dict[str, Any] = {"requested": float(target.calibrate_to.value), "achieved_auc": achieved, "scale": scale}
    else:
        scale = resolve_knob(target.link.scale, knob_rng)
        calibration = {"requested": None, "achieved_auc": None, "scale": scale}

    probability = probability_at(scale)
    caveat = caveat_seen
    calibration["intercept_shift"] = shift_seen
    if used_drivers:
        calibration["variance_drivers"] = used_drivers
    labels = sample_labels(probability, stream_for(spec.root_seed, spec.name, "labels", target.name))

    structural = build_ground_truth(spec, target_name=target.name, redundancy_groups=redundancy_groups)
    caveats = list(structural.caveats)
    if caveat:
        caveats.append(caveat)
    requested, achieved = calibration["requested"], calibration["achieved_auc"]
    if requested is not None and achieved is not None and abs(float(achieved) - float(requested)) > 0.01:
        caveats.append(f"requested ceiling AUC {float(requested):.3f} was not reachable; achieved {float(achieved):.3f}")
    # Rebuilt rather than mutated: both records are frozen, and reaching past that with object.__setattr__
    # would work today and break the moment either grows a cached derived field.
    truth = replace(
        structural,
        features={name: replace(entry, pre_standardization_scale=scales.get(name)) for name, entry in structural.features.items()},
        true_prob=probability,
        true_mean=probability,
        caveats=tuple(caveats),
    )

    calibration["bayes_auc"] = bayes_auc(probability)
    calibration["bayes_brier"] = bayes_brier(probability)
    calibration["prevalence"] = float(np.mean(probability))

    frame = _assemble_frame(spec, columns)
    return GeneratedDataset(frame=frame, target=pd.Series(labels, name=target.name), truth=truth, calibration=calibration)


def _probability_only(spec: DatasetSpec, target: TargetSpec, columns: Dict[str, np.ndarray], scale: float) -> np.ndarray:
    """Return the probabilities a given scale would produce, for callers exploring a difficulty sweep.

    Kept separate from :func:`generate` so a sweep does not redraw the data at every point: redrawing makes
    the difficulty curve non-monotone for a reason that has nothing to do with difficulty.
    """
    knob_rng = stream_for(spec.root_seed, spec.name, "knobs", target.name)
    score = scale * link_score(target.link, columns, int(spec.n_samples), knob_rng=knob_rng, scale_override=1.0)
    return probability_from_score(score, target.link.kind)
