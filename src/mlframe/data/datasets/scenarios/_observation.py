"""Beds where the structure is sound and the OBSERVATION of it is not: holes, imbalance, shift.

These share one property that makes them worth separating from every other family: the relationship
between the features and the target is untouched. The link is the same linear-logistic one the control bed
uses, the coefficients are the same, the calibrated ceiling is the same. What changes is what a method
gets to see of it.

That matters for how the results read. On a structural bed, a method failing has misunderstood the
dependence. Here, a method failing has mishandled the observation process, and the fix is a different one
entirely -- imputation, resampling, reweighting rather than a different scorer.

* **missingness** comes in three mechanisms that are different problems rather than degrees of one, and
  the bed carries all three on sibling columns so one run separates them. The recorded ceiling stays the
  COMPLETE-data ceiling, and the truth says so: the observed-data ceiling is lower by an amount this
  generator does not claim to compute.
* **imbalance** is set by moving the intercept, never by dropping majority rows. Dropping rows changes the
  sample size and the balance together, so a method that degrades with sample size would look like a
  method that degrades with imbalance. The rare-class bed is sized from the MINORITY count it needs -- a
  one-per-cent rate needs on the order of five thousand rows before anything measured on the minority is
  stable, and an undersized fixture reads as flakiness rather than as the undersizing it is.
* **shift** is a pair of beds sharing one structure: one moves ``P(x)`` and leaves ``P(y|x)`` alone, the
  other rotates ``P(y|x)`` and leaves ``P(x)`` alone. Anything that cannot tell them apart is reporting
  "drift" as one phenomenon when the two call for opposite responses.
"""

from __future__ import annotations

from typing import Dict, Tuple

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, GateSpec, LinkSpec, MissingnessSpec, TargetSpec

__all__ = ["missingness_trio_spec", "rare_class_spec", "covariate_shift_spec", "concept_shift_spec", "RARE_PREVALENCE", "RARE_MIN_ROWS"]

#: The rare-class rate. Low enough that the usual metrics misbehave, high enough that the bed is not a
#: test of whether anything at all survives.
RARE_PREVALENCE = 0.01

#: Rows the rare bed needs. Sized from the MINORITY count: at one per cent this yields roughly a hundred
#: positive rows, below which every metric computed on the minority is noise and the bed measures its own
#: undersizing.
RARE_MIN_ROWS = 10000


def _linear_core(n_informative: int, seed: int) -> Tuple[Tuple[FeatureSpec, ...], Dict[str, float]]:
    """Return the shared informative columns and their weights, identical across this family."""
    informative = tuple(FeatureSpec(name=f"s{i}") for i in range(n_informative))
    weights = {f"s{i}": float(1.2 * (0.72**i)) for i in range(n_informative)}
    return informative, weights


def missingness_trio_spec(n_noise: int = 25, n_samples: int = 6000, ceiling: float = 0.82, seed: int = 0, rate: float = 0.3) -> DatasetSpec:
    """Return the bed carrying all three missingness mechanisms at once, on sibling informative columns.

    One bed rather than three, because the comparison between mechanisms is the finding and running them
    separately would let the data draw differ between them. Every mechanism hides the same expected share
    of rows, so what differs is WHICH rows are hidden and never how many.

    * ``s0`` -- MCAR. The observed rows are a fair sample; only power is lost.
    * ``s1`` -- MAR, driven by ``s3``. Biased, but the bias is recoverable because the driver is observed.
    * ``s2`` -- MNAR, driven by its own value. Its largest values go missing BECAUSE they are large, and
      nothing observable identifies the bias.

    ``s3`` is left complete on purpose: it is what makes the MAR case recoverable, and a bed that hid it
    too would have turned its MAR column into a second MNAR one.
    """
    informative, weights = _linear_core(4, seed)
    return DatasetSpec(
        name=f"missingness_trio_{int(rate * 100)}pct",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes(n_noise),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        missingness=(
            MissingnessSpec(column="s0", mechanism="mcar", rate=rate),
            MissingnessSpec(column="s1", mechanism="mar", rate=rate, driver="s3"),
            MissingnessSpec(column="s2", mechanism="mnar", rate=rate),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "observation", "purpose": "three missingness mechanisms on sibling columns of one bed"},
    )


def rare_class_spec(n_noise: int = 30, n_samples: int = RARE_MIN_ROWS, ceiling: float = 0.85, seed: int = 0, prevalence: float = RARE_PREVALENCE) -> DatasetSpec:
    """Return the rare-positive bed: the same structure at a one-per-cent positive rate.

    Prevalence is reached by shifting the intercept, so every row and every feature is exactly what the
    balanced bed has. A bed built by dropping majority rows would differ in sample size as well as balance,
    and no comparison between the two could then attribute a difference to either.
    """
    informative, weights = _linear_core(5, seed)
    return DatasetSpec(
        name=f"rare_class_{int(prevalence * 1000):03d}permille",
        n_samples=max(int(n_samples), RARE_MIN_ROWS),
        root_seed=seed,
        features=informative + probes(n_noise),
        targets=(TargetSpec(name="y", prevalence=prevalence, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "observation", "purpose": "imbalance reached by intercept shift, so sample size is not confounded with balance"},
    )


def covariate_shift_spec(n_noise: int = 25, n_samples: int = 6000, ceiling: float = 0.82, seed: int = 0) -> DatasetSpec:
    """Return the covariate-shift half of the drift pair: ``P(x)`` moves, ``P(y|x)`` does not.

    The informative columns are drawn from a shifted, wider law while the link that turns them into a
    target is untouched. A model fitted here is still correct about the relationship; it has simply seen a
    different part of the input space, and the right response is reweighting rather than refitting.
    """
    informative = tuple(FeatureSpec(name=f"s{i}", family="normal", params={"loc": 0.8, "scale": 1.6}, standardize=False) for i in range(5))
    weights = {f"s{i}": float(1.2 * (0.72**i)) for i in range(5)}
    return DatasetSpec(
        name="shift_covariate",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes(n_noise),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "observation", "purpose": "P(x) moves and P(y|x) does not: the half of drift that reweighting fixes"},
    )


def concept_shift_spec(n_noise: int = 25, n_samples: int = 6000, ceiling: float = 0.82, seed: int = 0) -> DatasetSpec:
    """Return the concept-shift half of the drift pair: ``P(x)`` is unchanged, ``P(y|x)`` rotates.

    The columns are drawn exactly as the control bed draws them. What changes is the link: the effect is
    confined to a region of one column's range, so the relationship holds on part of the input space and
    is absent on the rest. A global summary statistic averages that away, which is the failure mode this
    half of the pair exists to expose -- and it is the half no amount of reweighting fixes.

    The two halves share a structure and differ in exactly one declaration, so a method that reports the
    same thing on both is reporting "drift" as one phenomenon when the two demand opposite responses.
    """
    informative, weights = _linear_core(5, seed)
    return DatasetSpec(
        name="shift_concept",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes(n_noise),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients=weights, region=GateSpec(column="s0", fraction=0.5)),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "observation", "purpose": "P(y|x) rotates while P(x) stays put: the half of drift reweighting cannot fix"},
    )
