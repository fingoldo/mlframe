"""Datasets and synthetic-data generation utilities.

Submodules:
    datasets    - loaders for built-in / common benchmark datasets.
    synthetic   - synthetic data generators for tabular ML scenarios.
"""

from __future__ import annotations


from mlframe.data.datasets import *
from mlframe.data.synthetic import *

# Curate the star-import surface explicitly (mirrors mlframe.metrics.__init__'s pattern).
__all__ = sorted(name for name in globals() if not name.startswith("_"))
