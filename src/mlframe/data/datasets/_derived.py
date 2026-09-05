"""Columns generated FROM other columns, so a declared graph is the data rather than a claim about it.

A spec's edges are not documentation. If a bed declares ``Y -> C <- S`` and then draws ``C`` as an
independent normal, the truth record says "collider" while the data says "noise": every structural test
passes -- the blanket is computed from the edges -- and the bed silently tests nothing. The spouse is
marginally invisible in that bed for the wrong reason, and a method that fails to find it fails a test that
was never administered.

This module closes that gap. Any observed column with parents in the graph is REALISED from those parents:
a weighted sum plus an independent residual, then standardised like every other column. Which parents are
available depends on where the column sits relative to the target, so the work splits in two:

* ancestors of the target are built BEFORE it, because the link consumes them;
* descendants of the target -- children like a collider, and their own descendants -- are built AFTER it,
  from the realised label.

A child of the label is built from the realised ``y`` rather than from ``true_prob`` on purpose: a real
downstream measurement reflects the outcome that happened, not the probability that produced it, and using
the probability would make the collider cleaner than any observable of its kind can be.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from mlframe.data.datasets._columns import standardize
from mlframe.data.datasets._rng import stream_for
from mlframe.data.datasets._scm import CausalGraph

logger = logging.getLogger(__name__)

__all__ = ["derived_order", "realize_derived"]

# Residual scale for a derived column, relative to its parents' contribution. Large enough that a child is
# not a deterministic copy of its parents -- which would make it an exact duplicate and change what the bed
# tests -- and small enough that the declared edge is the dominant term.
DEFAULT_RESIDUAL_SD = 0.6


def _feature_parents(graph: CausalGraph, node: str, observed: Set[str], target: str) -> Tuple[str, ...]:
    """Return the parents of ``node`` that are realisable columns or the target itself."""
    return tuple(parent for parent in graph.parents(node) if parent in observed or parent == target)


def derived_order(graph: CausalGraph, observed: Sequence[str], target: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return ``(before_target, after_target)``: derived columns in an order their parents precede them.

    Args:
        graph: The scenario's causal graph.
        observed: Observed column names.
        target: The target node's name.

    Returns:
        Two tuples of column names. The first holds columns the target's link depends on, the second holds
        columns that depend on the target.

    Raises:
        ValueError: If the derived columns cannot be ordered, which means a cycle survived the graph's own
            acyclicity check and every value produced afterwards would depend on evaluation order.
    """
    observed_set = set(observed)
    derived = [name for name in observed if _feature_parents(graph, name, observed_set, target)]
    after = {name for name in derived if target in graph.ancestors(name)}

    ordered: List[str] = []
    remaining = list(derived)
    while remaining:
        ready = [name for name in remaining if all(parent not in remaining for parent in _feature_parents(graph, name, observed_set, target))]
        if not ready:
            raise ValueError(f"cannot order derived columns {sorted(remaining)}: their parent relation has a cycle")
        ordered.extend(ready)
        remaining = [name for name in remaining if name not in set(ready)]

    return tuple(name for name in ordered if name not in after), tuple(name for name in ordered if name in after)


def realize_derived(
    names: Sequence[str],
    graph: CausalGraph,
    columns: Dict[str, np.ndarray],
    scales: Dict[str, float],
    target: str,
    target_values: Mapping[str, np.ndarray],
    root_seed: int,
    spec_name: str,
    residual_sd: float = DEFAULT_RESIDUAL_SD,
) -> None:
    """Overwrite each named column with a realisation of its declared parents, in place.

    ``columns`` and ``scales`` are mutated because the caller owns them and every consumer downstream reads
    the same dicts; returning copies would double the peak memory for no benefit on a frame this size.

    Args:
        names: Columns to realise, already in dependency order.
        graph: The causal graph.
        columns: Realised columns so far.
        scales: Pre-standardisation scales, updated for each column realised here.
        target: The target's name.
        target_values: Mapping holding the realised target under its own name; empty before the target
            exists, which is what makes a pre-target column with a target parent a spec error rather than a
            silent zero.
        root_seed: Dataset root seed.
        spec_name: Dataset name, namespacing the streams.
        residual_sd: Residual scale relative to the parents' contribution.

    Raises:
        KeyError: If a parent has not been realised yet, which means the ordering was bypassed.
    """
    observed = set(columns)
    for name in names:
        parents = _feature_parents(graph, name, observed, target)
        if not parents:
            continue
        contribution = np.zeros(len(columns[name]), dtype=np.float64)
        for parent in parents:
            if parent == target:
                if target not in target_values:
                    raise KeyError(f"column {name!r} declares the target {target!r} as a parent but the target is not realised yet")
                # Centred, so a child of the label carries the label's information without its base rate,
                # which would otherwise show up as an offset rather than as association.
                values = np.asarray(target_values[target], dtype=np.float64)
                contribution = contribution + (values - float(values.mean()))
            else:
                if parent not in columns:
                    raise KeyError(f"column {name!r} has parent {parent!r}, which has not been realised")
                weight = next((edge.weight for edge in graph.edges if edge.source == parent and edge.target == name), 1.0)
                contribution = contribution + float(weight) * columns[parent]
        residual = stream_for(root_seed, spec_name, "derived", name).normal(0.0, 1.0, contribution.shape[0])
        realised, scale = standardize(contribution + float(residual_sd) * residual)
        columns[name] = realised
        scales[name] = scale
        logger.debug("realised derived column %s from parents %s", name, parents)
