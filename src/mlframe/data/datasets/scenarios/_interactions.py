"""Synergy beds: parity, and parity next to a decoy that every marginal method prefers.

Parity is the case that separates methods rather than ranking them. Its operands have zero marginal
association with the target by construction -- flipping one operand's sign flips the answer regardless of
the others -- so no ranking built from one-column-at-a-time statistics can see them, however good the
statistic is.

The distinction from a product term is not pedantic and this repository has been caught by it twice. In
``x * y`` both operands stay marginally informative, so a "synergy" bed built from products tests nothing a
correlation ranking cannot do. Two builders here were named after parity while computing products; a name
guard now checks the property rather than the name.

``parity_plus_marginal_decoy`` adds the piece that makes the bed diagnostic rather than merely hard: a
column with genuine, strong marginal association that is NOT part of the parity structure. Without it, a
method that finds nothing and a method that finds the wrong thing score identically. With it, the two are
distinguishable, and what a marginal method does is visible: it takes the decoy and reports success.
"""

from __future__ import annotations

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["parity_spec", "parity_plus_decoy_spec"]


def parity_spec(order: int = 3, n_noise: int = 30, n_samples: int = 6000, ceiling: float = 0.85, seed: int = 0) -> DatasetSpec:
    """Return a pure parity bed of the given order.

    Args:
        order: How many operands the parity spans; two is already invisible marginally, three defeats
            pairwise search as well.
        n_noise: Independent probes.
        n_samples: Rows.
        ceiling: Achievable AUC to calibrate to.
        seed: Root seed.
    """
    if order < 2:
        raise ValueError(f"parity needs at least two operands; got {order}")
    operands = tuple(f"p{i}" for i in range(order))
    return DatasetSpec(
        name=f"xor{order}",
        n_samples=n_samples,
        root_seed=seed,
        features=(*(FeatureSpec(name=name) for name in operands), *probes(n_noise)),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(kind="parity", interactions=(operands,), interaction_weights=(1.0,)),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=name, target="y") for name in operands),
        provenance={"family": "interactions", "purpose": "operands are individually invisible"},
    )


def parity_plus_decoy_spec(order: int = 3, n_noise: int = 30, n_samples: int = 6000, decoy_weight: float = 0.9, seed: int = 0) -> DatasetSpec:
    """Return a parity bed with a marginally attractive column that is not part of the parity.

    The decoy is a genuine weak cause, not a red herring in the misleading sense: it really does move the
    target, just far less than the parity does. That is what makes the bed fair -- a method preferring it is
    not being tricked, it is revealing that its ranking cannot see anything else.
    """
    operands = tuple(f"p{i}" for i in range(order))
    return DatasetSpec(
        name=f"xor{order}_plus_marginal_decoy",
        n_samples=n_samples,
        root_seed=seed,
        features=(*(FeatureSpec(name=name) for name in operands), FeatureSpec(name="decoy"), *probes(n_noise)),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                link=LinkSpec(
                    kind="parity",
                    coefficients={"decoy": float(decoy_weight)},
                    interactions=(operands,),
                    interaction_weights=(2.0,),
                ),
                calibrate_to=CeilingTarget(metric="auc", value=0.85),
            ),
        ),
        edges=(*(EdgeSpec(source=name, target="y") for name in operands), EdgeSpec(source="decoy", target="y")),
        provenance={"family": "interactions", "purpose": "separates finding nothing from finding the wrong thing"},
    )
