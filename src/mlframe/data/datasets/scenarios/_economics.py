"""Beds where the right answer depends on what a column COSTS, and on how redundant it is.

Two families that the suite could express and never did.

**Cost.** `FeatureSpec.cost` has been in the spec since its first version, for a stated reason: the
cost/quality Pareto leg needs it and retrofitting a field changes every spec hash. Thirty-eight beds
later, not one varies it, so every Pareto figure this benchmark draws treats acquisition as free and
measures only model fits. That makes the frontier a statement about compute, which is the cheap half of
the real question -- a column that costs a laboratory test and a column that is already in the database
are not interchangeable at equal predictive value, and no bed here has ever said so.

**Graded redundancy.** The library has exact redundancy -- five near-copies at correlation 0.96 -- and
nothing between that and independence. A correlation threshold is the single most common knob in feature
selection and every method in the roster has one somewhere; a bed with one correlation level cannot show
where any of them breaks. This bed carries four clusters at 0.5, 0.8, 0.95 and 0.99, so the answer is a
curve rather than a point, and the point where a method's threshold bites is visible rather than assumed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LatentSpec, LinkSpec, TargetSpec

__all__ = ["expensive_signal_spec", "graded_redundancy_spec", "CORRELATION_LEVELS", "CHEAP_COST", "EXPENSIVE_COST"]

#: Within-cluster correlations the graded bed carries. Spread across the range every correlation-threshold
#: default in this repository sits in, so at least one level lands on each side of each default.
CORRELATION_LEVELS: Tuple[float, ...] = (0.50, 0.80, 0.95, 0.99)

#: What a column already in the database costs.
CHEAP_COST = 1.0

#: What a column that has to be acquired costs. Two orders of magnitude, because the interesting case is
#: not "slightly dearer" -- it is the one where a method that ignores cost picks a set nobody can afford.
EXPENSIVE_COST = 100.0


def _noise_for(correlation: float, loading: float = 1.0) -> float:
    """Return the residual scale that gives two reflections of one latent the requested correlation.

    Two reflections ``L*z + s*e`` correlate ``L^2 / (L^2 + s^2)``, so the scale follows in closed form.
    Solving for it rather than tuning by hand is what makes the declared level the delivered one: a bed
    whose "0.95 cluster" actually sits at 0.91 measures a different threshold than the one it names.
    """
    if not 0.0 < correlation < 1.0:
        raise ValueError(f"a within-cluster correlation must lie strictly inside (0, 1); got {correlation}")
    return float(loading) * math.sqrt((1.0 - correlation) / correlation)


def expensive_signal_spec(n_cheap_probes: int = 25, n_samples: int = 6000, ceiling: float = 0.82, seed: int = 0) -> DatasetSpec:
    """Return the bed where the informative columns are the expensive ones.

    Three expensive columns carry the signal, three cheap ones carry a weaker version of it, and the probes
    are free. A method selecting on predictive value alone takes the expensive set; a method that prices
    its selection can trade a little AUC for two orders of magnitude of cost.

    Neither answer is right in general, which is the point -- the bed exists so the Pareto figure has an
    axis that is about acquisition rather than about compute, and so "which arm is worth its cost" stops
    being a question about model fits alone.
    """
    expensive = tuple(FeatureSpec(name=f"lab{i}", cost=EXPENSIVE_COST) for i in range(3))
    cheap = tuple(FeatureSpec(name=f"free{i}", cost=CHEAP_COST) for i in range(3))
    weights: Dict[str, float] = {f"lab{i}": 1.2 for i in range(3)}
    weights.update({f"free{i}": 0.45 for i in range(3)})
    return DatasetSpec(
        name="expensive_signal",
        n_samples=n_samples,
        root_seed=seed,
        features=expensive + cheap + probes(n_cheap_probes),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "economics", "purpose": "the informative columns are the expensive ones, so a cost-blind selection is unaffordable"},
    )


def graded_redundancy_spec(n_noise: int = 20, n_samples: int = 6000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return four redundancy clusters at correlations 0.50, 0.80, 0.95 and 0.99.

    Each cluster is three reflections of its own latent, and exactly one member of each carries the
    signal. A correlation-threshold filter keeps whole clusters below its threshold and collapses them
    above it, so the level at which it starts collapsing is visible directly -- and so is what that costs,
    since collapsing a cluster whose members differ is not the same as collapsing one whose members do not.

    The residual scale of each cluster is solved for rather than tuned, so the level a cluster is named
    after is the level it delivers.
    """
    features: List[FeatureSpec] = []
    latents: List[LatentSpec] = []
    weights: Dict[str, float] = {}
    for level in CORRELATION_LEVELS:
        tag = f"c{round(level * 100):02d}"
        members = tuple(f"{tag}_{index}" for index in range(3))
        features.extend(FeatureSpec(name=name) for name in members)
        latents.append(LatentSpec(name=f"u_{tag}", reflections=members, loadings=(1.0, 1.0, 1.0), noise_sd=_noise_for(level)))
        # One member per cluster drives the target. The others are redundant with it to the declared
        # degree, which is what makes "did the method keep one representative" a scorable question.
        weights[members[0]] = 1.0

    return DatasetSpec(
        name="redundancy_graded",
        n_samples=n_samples,
        root_seed=seed,
        features=tuple(features) + probes(n_noise),
        latents=tuple(latents),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "economics", "purpose": "four correlation levels in one bed, so a threshold's breaking point is a curve rather than a guess"},
    )
