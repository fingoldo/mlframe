"""Building blocks shared by more than one scenario module."""

from __future__ import annotations

from typing import Tuple

from mlframe.data.datasets.spec import FeatureSpec

__all__ = ["probes"]


def probes(count: int) -> Tuple[FeatureSpec, ...]:
    """Return independent probe columns, named so they sort after nothing in particular and carry no signal."""
    return tuple(FeatureSpec(name=f"n{i:03d}") for i in range(count))
