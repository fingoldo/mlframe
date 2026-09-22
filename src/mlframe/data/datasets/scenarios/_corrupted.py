"""Beds whose LABELS are wrong, and the only place the declared ceiling and the reachable one diverge.

Every other bed in this library is clean: the achievable AUC is what the scenario declared, because the
calibrator bisected the link scale until it was. These beds are the case where that cannot happen. A bed
declaring "0.85 achievable" and then flipping fifteen per cent of its labels ships data whose real ceiling
is 0.843, and no link scale recovers the difference -- the information is gone from the labels.

That gap is the whole point, and it is why the corruption layer had to be able to state how it transforms
``true_prob`` before it was allowed to exist. With the transformation declared, the generator knows the
law the corrupted labels actually came from, so the ceiling stays exact and the loss is measurable rather
than merely suspected. Without it, an arm that "wins" on a noisy bed would be winning against a ceiling
nobody computed.

Three corruptions, chosen because they damage different things:

* **uniform flip** -- every row's label is flipped with the same probability. Symmetric, so the ranking is
  untouched in expectation and only the separability shrinks. The easy case, and the control for the other
  two.
* **feature-dependent flip** -- the flip rate is high inside a region of one column's range and zero
  outside it. The ranking IS damaged, unevenly, and the column that decides where is informative about the
  label's RELIABILITY while carrying no signal of its own. Nothing that scores a column by its association
  with the observed label can see that.
* **post-binning** -- the label is coarsened after the fact. Equivalent to observing a threshold of the
  truth rather than the truth, which is what happens whenever a continuous outcome reaches a dataset as a
  grade or a bucket.

The declared ceiling and the achieved one are both recorded, and the generator adds a caveat whenever they
differ by more than a percentage point. A report quoting the declared number on these beds is quoting a
number the data cannot reach.
"""

from __future__ import annotations

from typing import Tuple

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, GateSpec, LinkSpec, NoiseSpec, TargetSpec

__all__ = ["uniform_flip_spec", "feature_dependent_flip_spec", "coarsened_label_spec", "FLIP_RATE", "REGION_FLIP_RATE"]

#: Uniform flip rate. High enough that the ceiling visibly moves -- 15% costs about 0.007 AUC on this
#: structure -- and low enough that the bed is still solvable, since a bed nobody can solve ranks every
#: method equally and measures nothing.
FLIP_RATE = 0.15

#: Flip rate INSIDE the unreliable region. Much higher than the uniform rate on purpose: the region covers
#: less than a third of the rows, so a comparable amount of damage has to be concentrated to land.
REGION_FLIP_RATE = 0.35

#: Share of rows whose labels are unreliable in the feature-dependent bed.
UNRELIABLE_REGION = 0.3


def _structure(n_informative: int, n_noise: int) -> Tuple[Tuple[FeatureSpec, ...], dict]:
    """Return the informative columns and weights shared by all three beds in this family."""
    informative = tuple(FeatureSpec(name=f"s{i}") for i in range(n_informative))
    weights = {f"s{i}": float(1.3 * (0.75**i)) for i in range(n_informative)}
    return informative + probes(n_noise), weights


def uniform_flip_spec(n_informative: int = 5, n_noise: int = 30, n_samples: int = 6000, ceiling: float = 0.85, seed: int = 0, rate: float = FLIP_RATE) -> DatasetSpec:
    """Return the symmetric label-noise bed: every row flipped with the same probability.

    The control of the family. A uniform flip damages separability without touching the ranking in
    expectation, so a method that loses here loses to noise rather than to structure -- and the gap
    between the declared ceiling and the reachable one is the amount of information the flip destroyed.
    """
    features, weights = _structure(n_informative, n_noise)
    return DatasetSpec(
        name=f"label_flip_uniform_{int(rate * 100)}pct",
        n_samples=n_samples,
        root_seed=seed,
        features=features,
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients=weights),
                noise=NoiseSpec(kind="uniform_flip", rate=rate, true_prob_update="uniform_flip"),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "corrupted", "purpose": "symmetric label noise: the declared ceiling is not reachable and the gap is the damage"},
    )


def feature_dependent_flip_spec(n_informative: int = 5, n_noise: int = 30, n_samples: int = 6000, ceiling: float = 0.85, seed: int = 0, rate: float = REGION_FLIP_RATE) -> DatasetSpec:
    """Return the bed where label reliability depends on a column, and that column carries no signal itself.

    ``s0`` decides where the labels are unreliable: inside its top region they flip at a high rate, outside
    it they do not flip at all. It is informative about the label's RELIABILITY while carrying its own
    signal separately, which is a real and common shape -- a measurement device that is accurate only in
    part of its range, an annotator who is confident only on part of the input.

    Nothing that scores a column by its association with the OBSERVED label can see this. The damage is
    concentrated where a method is least able to notice it, and a method that reweights by an estimated
    reliability would do better than one that does not -- which is a distinction this suite could not draw
    on any other bed.
    """
    features, weights = _structure(n_informative, n_noise)
    return DatasetSpec(
        name=f"label_flip_regional_{int(rate * 100)}pct",
        n_samples=n_samples,
        root_seed=seed,
        features=features,
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients=weights),
                noise=NoiseSpec(kind="feature_dependent_flip", rate=rate, true_prob_update="feature_dependent_flip", gate=GateSpec(column="s0", fraction=UNRELIABLE_REGION)),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "corrupted", "purpose": "label reliability depends on a column, which no association-with-the-observed-label score can see"},
    )


def coarsened_label_spec(n_informative: int = 5, n_noise: int = 30, n_samples: int = 6000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return the bed whose label is a coarsened view of the truth rather than the truth.

    What happens whenever a continuous outcome reaches a dataset as a grade, a bucket or a threshold
    crossing: the label is a function of the truth, but a lossy one, and the loss is not noise -- it is
    systematic and identical for every row on the same side of the boundary.
    """
    features, weights = _structure(n_informative, n_noise)
    return DatasetSpec(
        name="label_coarsened",
        n_samples=n_samples,
        root_seed=seed,
        features=features,
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients=weights),
                noise=NoiseSpec(kind="binning", rate=0.1, true_prob_update="binning_pushforward"),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "corrupted", "purpose": "the label is a lossy function of the truth, not a noisy copy of it"},
    )
