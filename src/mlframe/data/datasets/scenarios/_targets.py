"""Beds whose target is not a coin flip, because a selector's job changes with the target.

Every other bed here is binary, which quietly restricts what the suite can conclude. Three kinds, sharing
one structure so that the only thing varying is the target:

* **multiclass** -- each class depends on a different rotation of the same weights, so the answer key is
  the UNION over classes. A filter that scores one column against one binarised label is answering a
  different question and will find one class's columns.
* **ordinal** -- the same weights, one latent axis, ordered cut points. The answer key is identical for
  every class, which is exactly what separates this from the case above: a method that does well here and
  badly there is failing on class STRUCTURE, not on class count.
* **count** -- a Poisson mean. Heteroscedastic by construction, since a Poisson's variance is its mean, so
  a scorer assuming constant variance is mis-weighted on the largest rows.

The three share their features, their weights and their seed. That is the whole design: any difference
between them is a difference in the target, and nothing else is free to vary.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["multiclass_spec", "ordinal_spec", "count_spec", "SHARED_WEIGHTS"]

#: The weights every bed in this family uses. Decaying, so each bed contains both an obvious signal and one
#: near the detection threshold; shared, so the three beds differ only in their target.
SHARED_WEIGHTS: Dict[str, float] = {"s0": 1.2, "s1": 0.9, "s2": 0.65, "s3": 0.45}


def _features(n_noise: int) -> Tuple[FeatureSpec, ...]:
    """Return the shared informative columns plus probes."""
    return tuple(FeatureSpec(name=name) for name in SHARED_WEIGHTS) + probes(n_noise)


def _bed(name: str, kind: Literal["binary", "multiclass", "ordinal", "count", "multilabel"], n_classes: int, n_noise: int, n_samples: int, seed: int, scale: float, purpose: str) -> DatasetSpec:
    """Build one bed of this family, varying only the target."""
    return DatasetSpec(
        name=name,
        n_samples=n_samples,
        root_seed=seed,
        features=_features(n_noise),
        targets=(TargetSpec(name="y", kind=kind, n_classes=n_classes, link=LinkSpec(kind="linear", coefficients=SHARED_WEIGHTS, scale=scale)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in SHARED_WEIGHTS),
        provenance={"family": "targets", "kind": kind, "purpose": purpose},
    )


def multiclass_spec(n_classes: int = 3, n_noise: int = 26, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the multiclass bed: each class depends on a different rotation of the shared weights."""
    return _bed(
        f"multiclass_k{n_classes}",
        "multiclass",
        n_classes,
        n_noise,
        n_samples,
        seed,
        scale=1.0,
        purpose="the answer key is the union over classes, so a one-versus-rest filter finds one class's columns",
    )


def ordinal_spec(n_classes: int = 4, n_noise: int = 26, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the ordinal bed: one latent axis and ordered cut points, same answer key for every class."""
    return _bed(
        f"ordinal_k{n_classes}",
        "ordinal",
        n_classes,
        n_noise,
        n_samples,
        seed,
        scale=1.0,
        purpose="ordered classes on one axis: a method treating them as unordered discards most of the signal",
    )


def count_spec(n_noise: int = 26, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the count bed: a Poisson mean, heteroscedastic because a Poisson's variance is its mean.

    The link scale is smaller than the other two beds use, and that is not a difficulty knob. An
    exponential mean turns a score of three into a mean of twenty, so the same scale that gives a sensible
    class balance gives counts in the hundreds; the scale here is what keeps the counts in the range a
    count column actually takes.
    """
    return _bed(
        "count_poisson",
        "count",
        2,
        n_noise,
        n_samples,
        seed,
        scale=0.35,
        purpose="variance grows with the mean, so a constant-variance scorer is mis-weighted on the largest rows",
    )
