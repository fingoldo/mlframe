"""Datasets and synthetic-data generation utilities.

Submodules:
    datasets    - loaders for built-in / common benchmark datasets.
    synthetic   - synthetic data generators for tabular ML scenarios.
"""

from __future__ import annotations


from mlframe.data.datasets import *
from mlframe.data.synthetic import *

# Literal, not derived from globals(): a comprehension over globals() also picks up this module's own imports (``annotations``)
# and anything a star-imported submodule leaks, so the curated surface silently drifts whenever a submodule adds an import.
__all__ = [
    "assign_classes_from_probability",
    "generate_modelling_data",
    "get_sapp_dataset",
    "indicator",
    "sample_random_variable",
    "showcase_pycaret_datasets",
]
