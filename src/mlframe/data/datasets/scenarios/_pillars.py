"""The four pillars of the feature-selection tests as beds: the published compound, pure noise, parity and conjunction.

Each builder reproduces one pillar's canonical instance from the test modules it was taken from, so a method's
behaviour here is the behaviour those tests were written to pin down, measured on a bed the arms were not
tuned against.

What is carried over and what is not, stated per pillar because the difference decides what a bed can say:

* **F2, the compound.** ``a**2 / b + log|c| * sin(d)`` over five operands on ``[1, 5]`` and ``[0, 2 pi]``, with
  ``e`` a pure noise column. The formula is reproduced exactly as the latent score. Its additive irreducible
  term ``f / 5`` is omitted: the bed's labels are drawn through the calibrated link, which already carries
  irreducible noise. Calibration bisects the link scale to the library's default ceiling, as the Friedman
  beds do, so the structure is the published one and the difficulty is the library's.
* **Pure noise.** Nine marginal kinds of eight columns each, against an independent binary target. The
  originals are per-kind frames; here they share one frame, so a method's false discoveries are counted
  across shapes at once. The original's two extra columns in the categorical and id-like frames are already
  present as normal columns in the mix, so they are not repeated. The AR(1) and random-walk noise of the
  four-pillar report is NOT here: those need a temporal bed the library does not have.
* **Three-way parity.** The same family as ``xor3``, at the test module's width of one hundred noise columns,
  so the bed differs from ``xor3`` by width alone.
* **Conjunction.** ``A, B, C`` Bernoulli(1/2), the logit ``-1.5 + 2.5 A + 3 A B C`` and one hundred probes.
  The original's proxy columns (fifty per operand, correlated with it at 0.7) are NOT here. They are the point
  of that test, and a proxy is a noisy reflection of a BINARY operand, which the reflection machinery does not
  express. Without them this is the conjunction core only, and the bed says so in its purpose.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import BasisTerm, CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec
from mlframe.data.datasets.scenarios._interactions import parity_spec

__all__ = [
    "pillar_f2_compound_spec",
    "pillar_noise_marginals_spec",
    "pillar_xor3_spec",
    "pillar_conjunction_spec",
    "NOISE_KINDS",
]

#: The nine marginal kinds of the pure-noise pillar, with the family each is drawn from. ``id`` is a permutation of
#: the row index, so every value is distinct; ``near_constant`` is zero on 99% of rows.
NOISE_KINDS: Tuple[Tuple[str, str, Dict[str, float]], ...] = (
    ("normal", "normal", {}),
    ("uniform", "uniform", {"low": -1.0, "high": 1.0}),
    ("student_t2", "student_t", {"df": 2.0}),
    ("lognormal", "lognormal", {"mean": 0.0, "sigma": 1.5}),
    ("exponential", "exponential", {"scale": 1.0}),
    ("binary", "bernoulli", {"p": 0.5}),
    ("categorical", "categorical", {"n_levels": 6.0}),
    ("near_constant", "zero_inflated_normal", {"loc": 0.0, "scale": 1.0, "inflation": 0.99}),
    ("id_like", "id", {}),
)

_PER_KIND = 8


def pillar_f2_compound_spec(n_samples: int = 3000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return the F2 compound: ``a**2 / b + log|c| sin(d)`` over five operands, with one pure noise column.

    Measured on the generated bed: ``c`` has no marginal association with the label (univariate AUC 0.50), and ``d``
    has a weak one of the opposite sign (AUC 0.46), which arises from the nonlinear link rather than from a linear
    term. A method that scores columns one at a time therefore finds ``a`` clearly, ``d`` weakly, and ``c`` not at all.
    """
    features = [FeatureSpec(name=name, family="uniform", params={"low": 1.0, "high": 5.0}, standardize=False) for name in ("a", "b", "c")]
    features.append(FeatureSpec(name="d", family="uniform", params={"low": 0.0, "high": 2.0 * math.pi}, standardize=False))
    features.append(FeatureSpec(name="e", family="uniform", params={"low": 1.0, "high": 5.0}, standardize=False))
    return DatasetSpec(
        name="pillar_f2_compound",
        n_samples=n_samples,
        root_seed=seed,
        features=tuple(features),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(
                    kind="logistic",
                    basis_terms=(
                        BasisTerm(kind="ratio", columns=("a", "b"), weight=1.0, params={"power": 2.0}),
                        BasisTerm(kind="log_abs_sin", columns=("c", "d"), weight=1.0),
                    ),
                ),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in ("a", "b", "c", "d")),
        provenance={"family": "pillar", "purpose": "the F2 compound a**2/b + log|c| sin(d), with its noise operand e"},
    )


def pillar_noise_marginals_spec(n_samples: int = 2000, seed: int = 0) -> DatasetSpec:
    """Return the pure-noise pillar: nine marginal shapes, eight columns each, against an independent binary target.

    Every column is independent of the target, so every selection is a false discovery. The shapes are the point:
    heavy tails and point masses are where a marginal test's nominal rate stops being its real one.
    """
    features = [
        FeatureSpec(name=f"{kind}_{j}", family=family, params=params, standardize=False) for kind, family, params in NOISE_KINDS for j in range(_PER_KIND)
    ]
    return DatasetSpec(
        name="pillar_noise_marginals",
        n_samples=n_samples,
        root_seed=seed,
        features=tuple(features),
        targets=(TargetSpec(name="y", prevalence=0.5, link=LinkSpec(kind="logistic", coefficients={})),),
        edges=(),
        provenance={"family": "pillar", "purpose": "pure noise across nine marginal shapes: a false discovery is a false discovery whatever the column's shape"},
    )


def pillar_xor3_spec(n_noise: int = 100, n_samples: int = 4000, seed: int = 0) -> DatasetSpec:
    """Return the three-way parity pillar: the ``xor3`` family at the pillar's width of one hundred noise columns."""
    base = parity_spec(order=3, n_noise=n_noise, n_samples=n_samples, seed=seed)
    return base.model_copy(update={"name": "pillar_xor3", "provenance": {"family": "pillar", "purpose": "three-way parity among one hundred noise columns"}})


def pillar_conjunction_spec(n_noise: int = 100, n_samples: int = 4000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return the conjunction core: ``logit = -1.5 + 2.5 A + 3 A B C`` over Bernoulli(1/2) operands and one hundred probes.

    The proxy columns of the original test are not included; see the module docstring.
    """
    binary = tuple(FeatureSpec(name=name, family="bernoulli", params={"p": 0.5}, standardize=False) for name in ("A", "B", "C"))
    return DatasetSpec(
        name="pillar_conjunction",
        n_samples=n_samples,
        root_seed=seed,
        features=(*binary, *probes(n_noise)),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(
                    kind="logistic",
                    coefficients={"A": 2.5},
                    interactions=(("A", "B", "C"),),
                    interaction_weights=(3.0,),
                    intercept=-1.5,
                ),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=name, target="y") for name in ("A", "B", "C")),
        provenance={"family": "pillar", "purpose": "conjunction of three operands among one hundred noise columns; the proxy decoys are not in this bed"},
    )
