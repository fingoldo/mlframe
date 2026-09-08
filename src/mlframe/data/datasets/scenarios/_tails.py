"""Beds whose signal lives in the joint tail, and the control that isolates the tail from the correlation.

An equal-mass binned mutual-information estimator with ten bins puts an entire joint tail into one cell of
the joint histogram. A dependence that exists ONLY there is therefore not merely hard for it to estimate --
it is invisible, and no amount of data fixes that, because the resolution is the problem. This repository's
own results already single out binned MI as the weakest family, so this is the bed that shows the specific
mechanism rather than the symptom.

The pair is what makes it an experiment rather than an anecdote. Both beds use the same columns, the same
marginals, the same link and the same rank correlation; they differ only in the copula. Under the Gaussian
copula there is no asymptotic tail dependence at all, so a method that fails on the t bed and succeeds on
the Gaussian one is failing on TAILS. A method that fails on both is failing on correlation, or on the gate,
and the tail claim would be unsupported.

Neither bed calibrates to a target ceiling. A gate that fires on a small joint region has whatever ceiling
it has, and bisecting the link scale to force a higher one would either widen the region -- destroying the
property being tested -- or fail and report the miss. The achieved ceiling travels with the bed instead.
"""

from __future__ import annotations

from typing import Literal, Tuple

from mlframe.data.datasets.spec import CopulaSpec, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["tail_dependence_spec", "gaussian_tail_control_spec", "TAIL_QUANTILE"]

#: Where the gate fires. High enough that the signal is genuinely a tail phenomenon, low enough that the
#: region holds a few hundred rows at the sizes these beds run at -- a gate nobody's sample reaches is not a
#: hard bed, it is an empty one.
TAIL_QUANTILE = 0.80


def _probes(count: int) -> Tuple[FeatureSpec, ...]:
    """Return independent probe columns."""
    return tuple(FeatureSpec(name=f"n{i:03d}") for i in range(count))


def _tail_bed(
    name: str,
    family: Literal["gaussian", "t", "clayton"],
    n_pair: int,
    n_noise: int,
    n_samples: int,
    seed: int,
    rho: float,
    df: float,
) -> DatasetSpec:
    """Build one tail bed under a named copula family."""
    pair = tuple(f"t{i}" for i in range(n_pair))
    return DatasetSpec(
        name=name,
        n_samples=n_samples,
        root_seed=seed,
        features=(*(FeatureSpec(name=column) for column in pair), *_probes(n_noise)),
        copulas=(CopulaSpec(columns=pair, family=family, rho=rho, df=df, margin="normal"),),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.35,
                link=LinkSpec(
                    kind="tail_gate",
                    interactions=(pair,),
                    interaction_weights=(1.0,),
                    tail_quantile=TAIL_QUANTILE,
                    scale=3.0,
                ),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in pair),
        provenance={"family": "tails", "copula": family},
    )


def tail_dependence_spec(n_pair: int = 2, n_noise: int = 30, n_samples: int = 6000, seed: int = 0, rho: float = 0.6, df: float = 4.0) -> DatasetSpec:
    """Return the t-copula bed: the columns co-occur in their tails, and only the joint tail drives the target."""
    return _tail_bed(f"tail_dependence_t{int(df)}", "t", n_pair, n_noise, n_samples, seed, rho, df)


def gaussian_tail_control_spec(n_pair: int = 2, n_noise: int = 30, n_samples: int = 6000, seed: int = 0, rho: float = 0.6) -> DatasetSpec:
    """Return the Gaussian-copula control: same correlation, same gate, no asymptotic tail dependence.

    Without this bed, a failure on the t bed is ambiguous between "cannot see tail dependence" and "cannot
    see a gate", which are different claims about a method and call for different fixes.
    """
    return _tail_bed("tail_control_gaussian", "gaussian", n_pair, n_noise, n_samples, seed, rho, 4.0)
