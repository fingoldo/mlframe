"""Causal beds: where the answer key, not the method, decides who wins.

These beds exist because "which columns should a selector pick" has three defensible answers, and on
ordinary data they coincide closely enough that nobody notices. Here they do not, and the disagreement is
the finding:

* **spouse-collider** (``Y -> C <- S``). The collider ``C`` is a child of the target, so it is trivially
  visible marginally and belongs in the Markov blanket. The spouse ``S`` is independent of the target until
  one conditions on ``C``, after which it becomes informative. So ``S`` is a canonical blanket member that
  no marginal method can reach, and ``C`` is a correct pick under prediction and a wrong one under causal
  discovery -- the same data, the same arms, opposite winners, decided entirely by an answer key nobody
  stated.
* **mediator chain** (``X -> M -> Y``, with a proxy of ``M``). Under the Markov blanket, ``M`` suffices and
  ``X`` is redundant; under causal parents, ``X`` is the cause and ``M`` is the mechanism. A benchmark that
  does not declare which it scores against is reporting a preference as a result.

The benchmark's pre-registered primary key is the Markov blanket, because feature selection here feeds
prediction and the blanket is the optimal predictive set and is unique. The other two are reported, never
silently substituted.
"""

from __future__ import annotations

from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["spouse_collider_spec", "mediator_chain_spec"]


def spouse_collider_spec(n_noise: int = 30, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the spouse-collider bed.

    The collider is generated as a function of both the target and the spouse. That construction is what
    makes the spouse reachable only after conditioning on the collider, and it is why the bed is built with
    an explicit edge from the target rather than by correlating columns until the numbers look right.
    """
    return DatasetSpec(
        name="mb_spouse_collider",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="x_cause"),
            FeatureSpec(name="spouse"),
            FeatureSpec(name="collider"),
            *(FeatureSpec(name=f"n{i:03d}") for i in range(n_noise)),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients={"x_cause": 1.0}),
                calibrate_to=CeilingTarget(metric="auc", value=0.78),
            ),
        ),
        edges=(
            EdgeSpec(source="x_cause", target="y"),
            EdgeSpec(source="y", target="collider"),
            EdgeSpec(source="spouse", target="collider"),
        ),
        provenance={"family": "causal", "purpose": "a blanket member unreachable by any marginal method"},
    )


def mediator_chain_spec(n_noise: int = 30, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the mediator-chain bed, with a proxy of the mediator alongside it.

    The proxy is cleaner than the mediator it reflects, which is the awkward part: scored against the
    Markov blanket it is a reasonable pick, scored against causal parents it is exactly the wrong one, and
    a method optimising prediction will prefer it precisely because it is cleaner.
    """
    return DatasetSpec(
        name="mediator_chain_with_proxy",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="x_cause"),
            FeatureSpec(name="mediator"),
            FeatureSpec(name="proxy"),
            *(FeatureSpec(name=f"n{i:03d}") for i in range(n_noise)),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.45,
                link=LinkSpec(kind="logistic", coefficients={"mediator": 1.0}),
                calibrate_to=CeilingTarget(metric="auc", value=0.80),
            ),
        ),
        edges=(
            EdgeSpec(source="x_cause", target="mediator"),
            EdgeSpec(source="mediator", target="y"),
            EdgeSpec(source="mediator", target="proxy", kind="proxy"),
        ),
        provenance={"family": "causal", "purpose": "the answer key, not the method, decides the winner"},
    )
