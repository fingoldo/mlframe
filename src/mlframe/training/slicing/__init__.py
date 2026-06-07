"""Slice-stable early-stopping subsystem.

Groups the slice-ES helpers used by the training loop:

- ``_slice_helpers`` -- shard / slice construction (``build_slice_eval_sets``
  producing ``SliceEvalSet`` shards over the validation set) + the
  ``effective_patience`` scaling for K-shard early stopping.
- ``_slice_pareto_plot`` -- the Pareto-frontier artifact
  (``generate_pareto_artifact``) rendered when slice-ES is configured to
  emit one.

The public surface is re-exported here so existing
``from mlframe.training.slicing import X`` import sites resolve from the
documented package path.
"""
from __future__ import annotations

from ._slice_helpers import (  # noqa: F401
    SliceEvalSet,
    build_slice_eval_sets,
    effective_patience,
)
from ._slice_pareto_plot import generate_pareto_artifact  # noqa: F401

__all__ = [
    "SliceEvalSet",
    "build_slice_eval_sets",
    "effective_patience",
    "generate_pareto_artifact",
]
