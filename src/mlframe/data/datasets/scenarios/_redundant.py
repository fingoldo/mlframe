"""Redundancy beds: the two cases the benchmark keeps confusing, built so they cannot be confused.

Correlated columns come in two kinds and they call for opposite behaviour:

* **Exact redundancy.** Several columns span one direction. One representative suffices, the rest are
  waste, and a method that keeps all of them is paying for nothing. Collapsing the cluster is CORRECT here.
* **Private-delta redundancy.** The columns share a latent factor but each also carries its own component,
  and the target is driven by an unequally weighted combination of those private components. The cluster
  mean no longer contains what drives the target, so collapsing the cluster DESTROYS signal that cannot be
  recovered afterwards.

Both are cluster-shaped and a correlation matrix cannot tell them apart, which is why a selector that
collapses clusters by correlation alone is right on the first and wrong on the second. Scoring the first by
"did it take one representative" and the second by "did it keep the members" is the whole point.

The unequal weighting is what makes the second bed real. Equal weights leave the cluster mean a sufficient
statistic for the driving combination, so aggregation costs nothing measurable while the docstring claims it
is destructive -- which is precisely the state an earlier version of this scenario shipped in.
"""

from __future__ import annotations

from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LatentSpec, LinkSpec, TargetSpec

__all__ = ["exact_redundancy_spec", "private_delta_spec"]


def exact_redundancy_spec(n_copies: int = 5, n_noise: int = 30, n_samples: int = 5000, correlation: float = 0.98, seed: int = 0) -> DatasetSpec:
    """Return a bed where one direction is observed through several near-identical columns.

    Collapsing this cluster is correct, so an arm is scored by whether it took ONE representative rather
    than by matching a member set: the members are interchangeable and requiring a particular one would fail
    a correct arm for an arbitrary preference.

    Args:
        n_copies: Columns spanning the shared direction.
        n_noise: Independent probes.
        n_samples: Rows.
        correlation: How tightly the copies track the latent; the residual noise is derived from it.
        seed: Root seed.
    """
    if not 0.0 < correlation < 1.0:
        raise ValueError(f"correlation must lie strictly inside (0, 1); got {correlation}")
    noise_sd = float((1.0 / correlation**2 - 1.0) ** 0.5)
    copies = tuple(f"c{i}" for i in range(n_copies))
    features = (*(FeatureSpec(name=name) for name in copies), *(FeatureSpec(name=f"n{i:03d}") for i in range(n_noise)))
    return DatasetSpec(
        name=f"redundant_exact_k{n_copies}_corr{int(correlation * 100)}",
        n_samples=n_samples,
        root_seed=seed,
        features=features,
        latents=(LatentSpec(name="z", reflections=copies, loadings=tuple(1.0 for _ in copies), distinct_sd=0.0, noise_sd=noise_sd),),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients={copies[0]: 1.0}),
                calibrate_to=CeilingTarget(metric="auc", value=0.80),
            ),
        ),
        edges=(*(EdgeSpec(source="z", target=name, kind="redundant_noisy") for name in copies), EdgeSpec(source=copies[0], target="y")),
        provenance={"family": "redundant", "purpose": "collapsing the cluster is CORRECT here"},
    )


def private_delta_spec(n_members: int = 3, n_noise: int = 30, n_samples: int = 5000, distinct_sd: float = 0.8, seed: int = 0) -> DatasetSpec:
    """Return a bed where the cluster members are jointly necessary and averaging them destroys the signal.

    The target is driven by the members' PRIVATE components with alternating, geometrically decaying
    weights. Any permutation-invariant weighting would make the cluster mean a sufficient statistic for the
    driving combination and quietly turn this bed into the exact-redundancy one.

    Args:
        n_members: Reflections of the shared latent.
        n_noise: Independent probes.
        n_samples: Rows.
        distinct_sd: Size of each member's private component; zero collapses this into exact redundancy.
        seed: Root seed.
    """
    members = tuple(f"r{i}" for i in range(n_members))
    features = (*(FeatureSpec(name=name) for name in members), *(FeatureSpec(name=f"n{i:03d}") for i in range(n_noise)))
    weights = {f"z::delta::{name}": float(((-1.0) ** i) * (1.0 / (1.6**i))) for i, name in enumerate(members)}
    return DatasetSpec(
        name=f"latent_replicates_private_delta_k{n_members}",
        n_samples=n_samples,
        root_seed=seed,
        features=features,
        latents=(LatentSpec(name="z", reflections=members, loadings=tuple(1.0 for _ in members), distinct_sd=distinct_sd),),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(kind="logistic", coefficients=weights),
                calibrate_to=CeilingTarget(metric="auc", value=0.80),
            ),
        ),
        edges=(
            *(EdgeSpec(source="z", target=name, kind="redundant_noisy") for name in members),
            *(EdgeSpec(source=name, target="y") for name in members),
        ),
        provenance={"family": "redundant", "purpose": "collapsing the cluster DESTROYS signal here"},
    )
