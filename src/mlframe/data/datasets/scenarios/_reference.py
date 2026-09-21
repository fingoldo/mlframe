"""Beds defined by published formulas, so at least part of this suite is comparable to somebody else.

Every other bed here was written by the author of one of the arms being judged. These were not. Friedman's
three functions (Friedman 1991, "Multivariate Adaptive Regression Splines", Annals of Statistics 19(1)) and
the Weston-Guyon design (Weston et al. 2003, "Use of the Zero-Norm with Linear Models and Kernel Methods",
JMLR 3) have been used as feature-selection beds for decades, which buys two things this suite cannot
generate for itself:

* **an outside check on the harness.** If a method recovers Friedman-1's five informative columns here and
  not in the literature, the harness is wrong, not the method. No self-written bed can produce that signal.
* **a bed nobody tuned against these arms.** The formulas predate the arms, so a surprising result on them
  cannot be an artefact of scenario choice -- the failure mode the pre-registration exists to bound.

Two honest caveats about what the port does and does not preserve:

* **Friedman's functions are regression targets and this suite's protocol is binary.** The formula is
  reproduced exactly as the latent score, and the binary label is drawn from it through the same calibrated
  link every other bed uses. So the STRUCTURE is the published one and the recovery task is the published
  one; the achievable AUC is not comparable to a published R-squared, and no table here claims it is.
* **Friedman-1 canonically carries five irrelevant columns.** That is too few to say anything about false
  discovery at this suite's widths, so the probe count is a parameter with the canonical value as its
  default and wider variants available by name. A bed run at a non-canonical width says so in its name.
"""

from __future__ import annotations

import math
from typing import Tuple

from mlframe.data.datasets.spec import BasisTerm, CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["friedman1_spec", "friedman2_spec", "friedman3_spec", "weston_guyon_spec"]


def _uniform_columns(count: int, prefix: str, low: float = 0.0, high: float = 1.0) -> Tuple[FeatureSpec, ...]:
    """Return uniform columns on ``[low, high]``, left UNSTANDARDIZED.

    Friedman's functions are defined on the unit hypercube and their constants -- the ``0.5`` centre of the
    squared term, the ``pi`` inside the sine -- are calibrated to that range. Standardising the columns to
    unit variance would move every one of those constants relative to the data and quietly produce a
    different function with the same name.
    """
    return tuple(FeatureSpec(name=f"{prefix}{i}", family="uniform", params={"low": low, "high": high}, standardize=False) for i in range(count))


def friedman1_spec(n_probes: int = 5, n_samples: int = 5000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return Friedman's first function: ``10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4``.

    Five informative columns spanning three shapes in one bed, which is what has kept it in use: a
    two-column non-monotone interaction, a symmetric quadratic no linear statistic can see, and two plain
    linear terms of unequal weight. A method can succeed on part of it and fail on the rest, and the answer
    key says which part.

    Args:
        n_probes: Irrelevant uniform columns. The canonical design uses five.
        n_samples: Rows.
        ceiling: Achievable AUC the link scale is calibrated against.
        seed: Root seed.

    Returns:
        The specification.
    """
    informative = _uniform_columns(5, "x")
    probes = _uniform_columns(n_probes, "p")
    canonical = n_probes == 5
    return DatasetSpec(
        name="friedman1" if canonical else f"friedman1_p{5 + n_probes}",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes,
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(
                    kind="linear",
                    coefficients={"x3": 10.0, "x4": 5.0},
                    basis_terms=(
                        BasisTerm(kind="sin_product", columns=("x0", "x1"), weight=10.0, params={"frequency": math.pi}),
                        BasisTerm(kind="centered_square", columns=("x2",), weight=20.0, params={"center": 0.5}),
                    ),
                ),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=f"x{i}", target="y") for i in range(5)),
        provenance={"family": "reference", "source": "Friedman 1991, Annals of Statistics 19(1)", "purpose": "three dependence shapes in one published bed"},
    )


def friedman2_spec(n_probes: int = 6, n_samples: int = 5000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return Friedman's second function, an impedance formula dominated by one ratio term.

    Published as ``sqrt(x0^2 + (x1 x2 - 1/(x1 x3))^2)`` over wildly different column ranges: ``x0`` in the
    hundreds, ``x3`` in the single digits. The ranges are the point. A method that standardises before
    scoring sees a different problem from one that does not, and this is the only bed here where that
    difference is built into the published design rather than added by this suite.

    The square root is omitted: it is monotone, so it changes no ranking, no recovery score and no
    selection -- and reproducing it would need a basis kind that exists for one bed.

    Args:
        n_probes: Irrelevant columns, drawn over the same span as the informative ones.
        n_samples: Rows.
        ceiling: Achievable AUC the link scale is calibrated against.
        seed: Root seed.

    Returns:
        The specification.
    """
    informative = (
        FeatureSpec(name="x0", family="uniform", params={"low": 0.0, "high": 100.0}, standardize=False),
        FeatureSpec(name="x1", family="uniform", params={"low": 40.0 * math.pi, "high": 560.0 * math.pi}, standardize=False),
        FeatureSpec(name="x2", family="uniform", params={"low": 0.0, "high": 1.0}, standardize=False),
        FeatureSpec(name="x3", family="uniform", params={"low": 1.0, "high": 11.0}, standardize=False),
    )
    probes = _uniform_columns(n_probes, "p", low=0.0, high=100.0)
    return DatasetSpec(
        name="friedman2" if n_probes == 6 else f"friedman2_p{4 + n_probes}",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes,
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(
                    kind="linear",
                    interactions=(("x1", "x2"),),
                    interaction_weights=(1.0,),
                    basis_terms=(BasisTerm(kind="ratio", columns=("x0", "x3"), weight=1.0, params={"floor": 1.0}),),
                ),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=name, target="y") for name in ("x0", "x1", "x2", "x3")),
        provenance={"family": "reference", "source": "Friedman 1991, Annals of Statistics 19(1)", "purpose": "unequal column ranges, published rather than invented"},
    )


def friedman3_spec(n_probes: int = 6, n_samples: int = 5000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return Friedman's third function, the same four columns in a ratio the arctangent compresses.

    Uses the same column ranges as :func:`friedman2_spec`, with the ratio as the whole signal rather than
    one term of it. The arctangent is monotone and therefore omitted for the same reason the square root is
    above: it changes the scale of the score and nothing about which columns matter.

    Args:
        n_probes: Irrelevant columns.
        n_samples: Rows.
        ceiling: Achievable AUC the link scale is calibrated against.
        seed: Root seed.

    Returns:
        The specification.
    """
    base = friedman2_spec(n_probes=n_probes, n_samples=n_samples, ceiling=ceiling, seed=seed)
    link = LinkSpec(
        kind="linear",
        basis_terms=(BasisTerm(kind="ratio", columns=("x1", "x0"), weight=1.0, params={"floor": 1.0}),),
        interactions=(("x1", "x2"),),
        interaction_weights=(-1.0,),
    )
    target = TargetSpec(name="y", prevalence=0.5, link=link, calibrate_to=CeilingTarget(metric="auc", value=ceiling))
    return base.model_copy(
        update={
            "name": "friedman3" if n_probes == 6 else f"friedman3_p{4 + n_probes}",
            "targets": (target,),
            "provenance": {"family": "reference", "source": "Friedman 1991, Annals of Statistics 19(1)", "purpose": "a ratio as the entire signal"},
        }
    )


def weston_guyon_spec(n_probes: int = 96, n_samples: int = 5000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return the Weston-Guyon linear design: a handful of informative columns against many probes.

    The design that the zero-norm and SVM-RFE literature is reported on: a small informative set with
    equal weights, buried in probes drawn from the same marginal so that nothing but the target
    distinguishes them. Equal weights matter -- a decaying weight vector lets a method score well by
    finding only the loudest column, and this design deliberately refuses that partial credit.

    It is the closest thing here to a pure false-discovery bed with a non-empty answer key, which is why
    the arms it expects to defeat are the ones that select generously.

    Args:
        n_probes: Irrelevant columns drawn from the same standard normal as the informative ones.
        n_samples: Rows.
        ceiling: Achievable AUC the link scale is calibrated against.
        seed: Root seed.

    Returns:
        The specification.
    """
    n_informative = 4
    informative = tuple(FeatureSpec(name=f"s{i}") for i in range(n_informative))
    probes = tuple(FeatureSpec(name=f"n{i:03d}") for i in range(n_probes))
    weights = {f"s{i}": 1.0 for i in range(n_informative)}
    return DatasetSpec(
        name=f"weston_guyon_k{n_informative}_p{n_informative + n_probes}",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes,
        targets=(TargetSpec(name="y", prevalence=0.5, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=name, target="y") for name in weights),
        provenance={"family": "reference", "source": "Weston et al. 2003, JMLR 3", "purpose": "equal-weight informative set against same-marginal probes"},
    )
