"""Beds whose signal lives in the joint tail, and the control that isolates the tail from the correlation.

The signal fires only where both columns sit in the SAME tail, upper or lower. Symmetry is what removes the
marginal channel: being high, on its own, is then no more predictive than being low, so a correlation or a
t-statistic sees essentially nothing (+0.01) and what is left is genuinely joint.

The first version of these beds gated on the upper tail alone, and it did not test what it claimed. Being
high is necessary for an upper-only gate, so each column carried a marginal correlation of +0.51 with the
target and every univariate filter recovered the pair perfectly. That bed's prediction -- that binned mutual
information would fail here -- was scored and FAILED, which is the pre-registration working; the fix is a
construction that matches the stated purpose, not a re-scoring of the old one.

What these beds do NOT show, and the earlier draft of this docstring wrongly claimed they would: binned MI is
attenuated here, not blind. With ten equal-mass bins a single column still carries roughly a third of the
gate's own mutual information, because the extreme bins stay enriched once the two directions cancel.

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

from typing import Literal

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CopulaSpec, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = [
    "tail_dependence_spec",
    "gaussian_tail_control_spec",
    "tail_isolation_spec",
    "TAIL_QUANTILE",
    "ISOLATION_QUANTILE",
    "ISOLATION_THETA",
    "ISOLATION_MATCHED_RHO",
    "ISOLATION_PAIRS",
]

#: Where the gate fires. High enough that the signal is genuinely a tail phenomenon, low enough that the
#: region holds a few hundred rows at the sizes these beds run at -- a gate nobody's sample reaches is not a
#: hard bed, it is an empty one.
TAIL_QUANTILE = 0.80


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
        features=(*(FeatureSpec(name=column) for column in pair), *probes(n_noise)),
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
    """Return the t-copula bed: the columns co-occur in their tails, and only the joint tails drive the target."""
    return _tail_bed(f"joint_tail_t{int(df)}", "t", n_pair, n_noise, n_samples, seed, rho, df)


def gaussian_tail_control_spec(n_pair: int = 2, n_noise: int = 30, n_samples: int = 6000, seed: int = 0, rho: float = 0.6) -> DatasetSpec:
    """Return the Gaussian-copula control: same correlation, same gate, no asymptotic tail dependence.

    Without this bed, a failure on the t bed is ambiguous between "cannot see tail dependence" and "cannot
    see a gate", which are different claims about a method and call for different fixes.
    """
    return _tail_bed("joint_tail_gaussian_control", "gaussian", n_pair, n_noise, n_samples, seed, rho, 4.0)


#: Where the isolation bed's one-sided gate fires. Deep, because that is the only place the two copulas
#: differ by enough to measure: at the tenth percentile a Clayton pair and a Spearman-matched Gaussian pair
#: co-occur 1.65 times as often, at the second percentile three times as often (measured, n = 1e6).
ISOLATION_QUANTILE = 0.02

#: Clayton dependence parameter. Deliberately the WEAKEST of the values measured: the contrast between the
#: copulas shrinks as the dependence strengthens (ratio 4.0 at theta=1, 1.4 at theta=8), because a strongly
#: dependent Gaussian pair also co-occurs in the tail. The bed wants the widest gap, not the strongest
#: dependence.
ISOLATION_THETA = 1.0

#: The Gaussian correlation whose Spearman rank correlation matches Clayton's at :data:`ISOLATION_THETA`,
#: via ``rho = 2 sin(pi rho_s / 6)``. Matching on RANK correlation rather than on linear correlation is what
#: makes the control a control: the two pairs are then indistinguishable to every rank statistic, and only
#: the depth of the tail tells them apart.
ISOLATION_MATCHED_RHO = 0.495


#: How many independent pairs of each copula family the bed carries. One pair of each is the clean design
#: and it produced a bed with an achievable AUC of 0.516 -- a gate at the second percentile fires on about
#: one row in a hundred, so a single pair simply has too little mass to drive a target. Stacking independent
#: pairs raises the signal WITHOUT making any individual pair shallower, which is the one axis that does not
#: trade the contrast away: four pairs of each family reach a usable ceiling while every pair keeps the
#: threefold firing-rate gap that the bed exists to measure.
ISOLATION_PAIRS = 4


def tail_isolation_spec(n_noise: int = 24, n_samples: int = 60000, seed: int = 0) -> DatasetSpec:
    """Return the bed that isolates tail dependence from everything correlated with it.

    The existing t-copula bed and its Gaussian control separate the roster IDENTICALLY, which means they
    demonstrate non-monotonicity and say nothing about tails. The reason is measurable: a symmetric gate at
    the eightieth percentile fires only about 1.1 times as often under a t copula as under a correlation-
    matched Gaussian one, so the two beds really are almost the same bed.

    This one is built to make the copula the ONLY difference that matters:

    * **matched pairs, one bed.** ``c<i>a``/``c<i>b`` are Clayton, ``g<i>a``/``g<i>b`` Gaussian at the
      correlation whose Spearman matches Clayton's. Same marginals, same rank correlation, same weight in
      the link, four independent pairs of each.
    * **a one-sided gate.** Clayton's dependence is entirely in the LOWER tail; a symmetric gate would add
      an upper tail where it is asymptotically independent and halve the contrast.
    * **deep.** At the second percentile the Clayton pair fires three times as often as its control, so it
      carries three times the signal despite being declared identically.

    Every one of these columns is a genuine cause of the target, so the answer key is all of them and set recovery is
    NOT the discriminating measurement here. The measurement is the RANK GAP: does a method rank the
    tail-dependent pair above the correlation-matched one? A method reading rank correlation, linear
    correlation or a ten-bin mutual information cannot -- those statistics are equal for the two pairs by
    construction. One that resolves the deep tail can.

    Args:
        n_noise: Probe columns.
        n_samples: Rows. Large by this suite's standards on purpose: a gate at the second percentile fires
            on roughly 1% of rows for the signal pair, so a smaller bed measures counting noise.
        seed: Root seed.

    Returns:
        The specification.
    """
    clayton = tuple((f"c{i}a", f"c{i}b") for i in range(ISOLATION_PAIRS))
    gaussian = tuple((f"g{i}a", f"g{i}b") for i in range(ISOLATION_PAIRS))
    columns = tuple(name for pair in clayton + gaussian for name in pair)
    return DatasetSpec(
        name="tail_isolation_clayton_vs_gaussian",
        n_samples=n_samples,
        root_seed=seed,
        features=(*(FeatureSpec(name=column) for column in columns), *probes(n_noise)),
        copulas=(
            *(CopulaSpec(columns=pair, family="clayton", theta=ISOLATION_THETA, margin="normal") for pair in clayton),
            *(CopulaSpec(columns=pair, family="gaussian", rho=ISOLATION_MATCHED_RHO, margin="normal") for pair in gaussian),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.35,
                link=LinkSpec(
                    kind="tail_gate",
                    interactions=clayton + gaussian,
                    # Equal weights: the whole design is that the two terms are declared identically and
                    # differ only in how often the data lets them fire.
                    interaction_weights=(1.0,) * (2 * ISOLATION_PAIRS),
                    tail_quantile=1.0 - ISOLATION_QUANTILE,
                    tail_direction="lower",
                    scale=3.0,
                ),
            ),
        ),
        edges=tuple(EdgeSpec(source=column, target="y") for column in columns),
        provenance={"family": "tails", "copula": "clayton+gaussian", "purpose": "tail dependence isolated from rank correlation"},
    )
