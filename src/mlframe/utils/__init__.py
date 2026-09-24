"""Miscellaneous helpers.

Submodules:
    eda          - exploratory-data-analysis re-exports.
    experiments  - experiment-tracking helpers.
    text         - text preprocessing utilities.
    misc         - small generic helpers (get_pipeline_last_element, ...).
    log_throttle - per-call-site log throttling for hot loops.

Every name is re-exported LAZILY (PEP 562). The package used to star-import all of the above at import time, so the
171 modules that import the 27 ms leaf ``mlframe.utils.log_throttle`` also paid ``experiments`` -> ``pyutilz.db`` ->
SQLAlchemy and redis: measured 3.64 s in a cold process, against 0.027 s for the leaf module on its own. Attribute
access resolves the owning submodule on first use, so a process that only throttles logs never imports SQLAlchemy.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Eager, and the only one: it is the 27 ms leaf this whole indirection exists to keep cheap, and binding the FUNCTION
# here is what keeps ``from mlframe.utils import log_throttle`` giving the function rather than its module (Python
# binds the submodule under that same name as soon as anything imports it).
from mlframe.utils.log_throttle import log_throttle

# name -> submodule that defines it. Kept explicit: building it by importing the submodules would reinstate the very
# eager imports this indirection exists to avoid.
_EXPORTS: dict[str, str] = {
    "showcase_df_columns": "mlframe.utils.eda",
    "create_experiment": "mlframe.utils.experiments",
    "get_experiments": "mlframe.utils.experiments",
    "get_experiment_routes": "mlframe.utils.experiments",
    "read_experiment": "mlframe.utils.experiments",
    "read_route": "mlframe.utils.experiments",
    "update_routes_audiences": "mlframe.utils.experiments",
    "read_glove_embeddings": "mlframe.utils.text",
    "set_random_seed": "mlframe.utils.misc",
    "set_numba_random_seed": "mlframe.utils.misc",
    "rng_hygienic_fit": "mlframe.utils.misc",
    "hygienic_fit": "mlframe.utils.misc",
    "preserve_global_rng": "mlframe.utils.misc",
    "get_pipeline_last_element": "mlframe.utils.misc",
    "get_full_classifier_name": "mlframe.utils.misc",
    "is_cuda_available": "mlframe.utils.misc",
    "check_cpu_flag": "mlframe.utils.misc",
    # The param-oracle public surface lives here so cross-package consumers
    # (``feature_selection.filters._meta_fe_recommender``, the recommender CLI) import it from ``mlframe.utils``
    # instead of reaching into ``mlframe.utils._param_oracle``, which the underscore-import meta-linter flags.
    "ParamOracle": "mlframe.utils._param_oracle",
    "bucketize_fingerprint": "mlframe.utils._param_oracle",
    "default_fingerprint": "mlframe.utils._param_oracle",
    "loads_json": "mlframe.utils._param_oracle",
    "stable_json": "mlframe.utils._param_oracle",
}

__all__ = sorted([*_EXPORTS, "log_throttle"])


def __getattr__(name: str):
    """Import the submodule that owns ``name`` on first access and cache the attribute on this module (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # subsequent lookups skip __getattr__ entirely
    return value


def __dir__() -> list:
    """The lazy names plus whatever has already been resolved, so tab-completion and ``dir()`` still show the surface."""
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # the names above, for type checkers and IDEs, without importing anything at runtime
    from mlframe.utils._param_oracle import ParamOracle, bucketize_fingerprint, default_fingerprint, loads_json, stable_json
    from mlframe.utils.eda import showcase_df_columns
    from mlframe.utils.experiments import (
        create_experiment,
        get_experiment_routes,
        get_experiments,
        read_experiment,
        read_route,
        update_routes_audiences,
    )
    from mlframe.utils.misc import (
        check_cpu_flag,
        get_full_classifier_name,
        get_pipeline_last_element,
        hygienic_fit,
        is_cuda_available,
        preserve_global_rng,
        rng_hygienic_fit,
        set_numba_random_seed,
        set_random_seed,
    )
    from mlframe.utils.text import read_glove_embeddings
