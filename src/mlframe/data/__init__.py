"""Datasets and synthetic-data generation utilities.

Submodules:
    datasets    - loaders for built-in / common benchmark datasets, plus the SCM-first synthetic core.
    synthetic   - synthetic data generators for tabular ML scenarios.

Imports are explicit rather than star. Two reasons, both learned here: a star import binds whatever the
submodule advertises, so two of them in sequence silently shadow each other on any shared name; and a star
import over a package that resolves names through PEP 562 forces every one of them, defeating the laziness
that keeps `import mlframe.data` cheap. The SCM core (`DatasetSpec`, `GroundTruth`, the graph helpers) is
reached by path -- `from mlframe.data.datasets.spec import DatasetSpec` -- or as an attribute of
`mlframe.data.datasets`, which resolves it on first use.
"""

from __future__ import annotations

from mlframe.data.datasets import get_sapp_dataset, indicator, showcase_pycaret_datasets
from mlframe.data.synthetic import assign_classes_from_probability, generate_modelling_data, sample_random_variable

__all__ = [
    "assign_classes_from_probability",
    "generate_modelling_data",
    "get_sapp_dataset",
    "indicator",
    "sample_random_variable",
    "showcase_pycaret_datasets",
]
