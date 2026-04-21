"""Public surface for mlframe.feature_selection.

The individual submodules are the source of truth; this file curates what
downstream code should depend on so internal refactors (e.g. splitting
``filters.py``) don't break imports silently.
"""

from mlframe.feature_selection.general import (
    estimate_features_relevancy,
    run_efs,
    benchmark_mi_algos,
)
from mlframe.feature_selection.mi import (
    grok_compute_mutual_information,
    chatgpt_compute_mutual_information,
    deepseek_compute_mutual_information,
)
from mlframe.feature_selection.wrappers import (
    RFECV,
    OptimumSearch,
    VotesAggregation,
)

__all__ = [
    # general
    "estimate_features_relevancy",
    "run_efs",
    "benchmark_mi_algos",
    # mi kernels
    "grok_compute_mutual_information",
    "chatgpt_compute_mutual_information",
    "deepseek_compute_mutual_information",
    # wrappers
    "RFECV",
    "OptimumSearch",
    "VotesAggregation",
]
