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
from mlframe.feature_selection.structure_discovery import (
    discover_structure,
    StructureReport,
    DiscoveredRelation,
)
from mlframe.feature_selection.ace import ace_select, ACEResult
from mlframe.feature_selection.forward_select import forward_select
from mlframe.feature_selection.cascade_select import cascade_select
from mlframe.feature_selection.ridge_forward_prefilter import ridge_coefficient_prefilter
from mlframe.feature_selection.unanimous_permutation_prune import unanimous_permutation_prune

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
    # structure discovery / EDA: surface hidden discrete relationships (gcd / modular / regime-switch / argmax) in (X, y)
    "discover_structure",
    "StructureReport",
    "DiscoveredRelation",
    # artificial-contrast feature significance (parametric t-test vs permuted-contrast importances + masking loop)
    "ace_select",
    "ACEResult",
    # forward selection + 3-stage cascade (Boruta screen -> forward select -> permutation backward elimination)
    "forward_select",
    "cascade_select",
    # cheap Ridge-coefficient fast pre-filter ahead of MRMR/RFECV, for thousands of raw candidate features
    "ridge_coefficient_prefilter",
    # strict all-folds-must-agree permutation-importance pruning
    "unanimous_permutation_prune",
]
