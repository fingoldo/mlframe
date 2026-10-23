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
  four-pillar report is carried by the two temporal beds (``pillar_noise_ar1``, ``pillar_noise_random_walk``), whose
  columns and target share one serially dependent family.
* **Three-way parity.** The same family as ``xor3``, at the test module's width of one hundred noise columns,
  so the bed differs from ``xor3`` by width alone.
* **Conjunction.** ``A, B, C`` Bernoulli(1/2), the logit ``-1.5 + 2.5 A + 3 A B C`` and one hundred probes, plus
  fifty proxies per operand. Each proxy is a derived child of its operand at correlation 0.7, so it is a decoy
  with no path to the label except through its operand.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from mlframe.data.datasets._derived import DEFAULT_RESIDUAL_SD
from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import BasisTerm, CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LatentSpec, LinkSpec, TargetSpec
from mlframe.data.datasets.scenarios._interactions import parity_spec

__all__ = [
    "pillar_f2_compound_spec",
    "pillar_noise_marginals_spec",
    "pillar_xor3_spec",
    "pillar_conjunction_spec",
    "pillar_noise_ar1_spec",
    "pillar_noise_random_walk_spec",
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


def _temporal_noise_spec(name: str, family: str, n_samples: int, seed: int, purpose: str) -> DatasetSpec:
    """Return a pure-noise bed whose columns and target are both serially dependent, from one column family.

    The target is a latent factor of the same family, which no column reads: the label is a time series of its own, so
    every association between a column and the label is spurious, which is the setting the test is about. Both halves of the
    pair are needed, because spurious regression needs two unrelated series that both wander.
    """
    if family == "ar1":
        features = [FeatureSpec(name=f"x{j}", family="ar1", params={"rho": 0.9}, standardize=False) for j in range(8)]
    else:
        features = [FeatureSpec(name=f"x{j}", family="random_walk", standardize=False) for j in range(8)]
    return DatasetSpec(
        name=name,
        n_samples=n_samples,
        root_seed=seed,
        features=tuple(features),
        latents=(LatentSpec(name="u", family=family),),
        targets=(TargetSpec(name="y", prevalence=0.5, link=LinkSpec(kind="logistic", coefficients={"u": 5.0})),),
        edges=(EdgeSpec(source="u", target="y", kind="latent"),),
        provenance={"family": "pillar", "purpose": purpose},
    )


def pillar_noise_ar1_spec(n_samples: int = 2000, seed: int = 0) -> DatasetSpec:
    """Return the AR(1) half of the time-series noise pillar: rho 0.9 columns, an AR(1) target, no link between them."""
    return _temporal_noise_spec(
        "pillar_noise_ar1", "ar1", n_samples, seed, "stationary AR(1) noise against an AR(1) target: the iid permutation null is anti-conservative"
    )


def pillar_noise_random_walk_spec(n_samples: int = 2000, seed: int = 0) -> DatasetSpec:
    """Return the random-walk half of the time-series noise pillar: walk columns against a walk target, no link."""
    return _temporal_noise_spec(
        "pillar_noise_random_walk", "random_walk", n_samples, seed, "independent random walks against a random-walk target: spurious regression"
    )


def pillar_xor3_spec(n_noise: int = 100, n_samples: int = 4000, seed: int = 0) -> DatasetSpec:
    """Return the three-way parity pillar: the ``xor3`` family at the pillar's width of one hundred noise columns."""
    base = parity_spec(order=3, n_noise=n_noise, n_samples=n_samples, seed=seed)
    return base.model_copy(update={"name": "pillar_xor3", "provenance": {"family": "pillar", "purpose": "three-way parity among one hundred noise columns"}})


#: Proxies per operand, and the correlation each has with its operand, as in the conjunction test's canonical frame.
N_PROXY = 50
PROXY_RHO = 0.7


def _proxy_weight(rho: float = PROXY_RHO, operand_sd: float = 0.5) -> float:
    """Return the edge weight that makes a derived proxy correlate ``rho`` with a binary operand of sd ``operand_sd``.

    The derived layer builds a child as the standardised sum of its weighted parent and a residual of scale
    ``DEFAULT_RESIDUAL_SD``. For a child ``w * x + s * e`` the correlation with ``x`` is ``w sd / sqrt(w^2 sd^2 + s^2)``,
    which solves in closed form for ``w``. Solving rather than tuning keeps the declared 0.7 the delivered 0.7.
    """
    s = float(DEFAULT_RESIDUAL_SD)
    return float(rho * s / (operand_sd * math.sqrt(1.0 - rho * rho)))


def pillar_conjunction_spec(n_noise: int = 100, n_samples: int = 50_000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return the conjunction pillar: ``logit = -1.5 + 2.5 A + 3 A B C``, fifty proxies per operand, and one hundred probes.

    Each proxy is a derived copy of its operand with correlation 0.7, so the decoys are real structure in the graph: a
    proxy is a child of its operand and of nothing else. At this width a fixed-K method takes a budget that the fifty
    proxies of A nearly exhaust, which is the mechanism the bed is built to test.
    """
    binary = tuple(FeatureSpec(name=name, family="bernoulli", params={"p": 0.5}, standardize=False) for name in ("A", "B", "C"))
    proxies = tuple(FeatureSpec(name=f"dec{op}_{k}") for op in ("A", "B", "C") for k in range(N_PROXY))
    proxy_edges = tuple(EdgeSpec(source=op, target=f"dec{op}_{k}", kind="proxy", weight=_proxy_weight()) for op in ("A", "B", "C") for k in range(N_PROXY))
    return DatasetSpec(
        name="pillar_conjunction",
        n_samples=n_samples,
        root_seed=seed,
        features=(*binary, *proxies, *probes(n_noise)),
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
        edges=(*(EdgeSpec(source=name, target="y") for name in ("A", "B", "C")), *proxy_edges),
        provenance={"family": "pillar", "purpose": "conjunction of three operands with fifty correlated proxies each, among one hundred probes"},
    )
