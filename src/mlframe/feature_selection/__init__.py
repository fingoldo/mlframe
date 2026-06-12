"""Public surface for mlframe.feature_selection.

The individual submodules are the source of truth; this file curates what
downstream code should depend on so internal refactors (e.g. splitting
``filters.py``) don't break imports silently.
"""

from __future__ import annotations


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
from mlframe.feature_selection.hybrid_selector import HybridSelector
from mlframe.feature_selection.compare_selectors import (
    compare_selectors,
    SelectorComparison,
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
    # hybrid (compute-once-share-many composition of MRMR/RFECV/BorutaShap/ShapProxiedFS)
    "HybridSelector",
    # read-only diagnostics: compare what each selector keeps (agreement / Jaccard / consensus)
    "compare_selectors",
    "SelectorComparison",
]
