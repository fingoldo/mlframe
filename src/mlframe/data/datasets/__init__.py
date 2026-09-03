"""Synthetic and showcase dataset generators used for benchmarking mlframe's feature-selection and training pipelines."""

from __future__ import annotations

from mlframe.data.datasets._sapp import get_sapp_dataset, indicator
from mlframe.data.datasets._showcase import showcase_pycaret_datasets

# Literal list of string constants: vulture only recognises this exact form as a re-export marker,
# and mlframe.data star-imports this package, so an implicit surface would silently shadow mlframe.data.synthetic.
__all__ = [
    "get_sapp_dataset",
    "indicator",
    "showcase_pycaret_datasets",
]
