"""Ensemble blending/weighting/pruning facade: hill-climb, NNLS/geometric/constrained blends, Shapley weighting, and related diagnostics.

Re-exports are LAZY (PEP 562) and the surface is declared in ``__all__``.

Previously all fifteen names were imported eagerly with no ``__all__``, which cost every consumer of
``mlframe.votenrank`` the import of fifteen submodules -- and their transitive scipy / sklearn / numba
dependencies -- when in practice only ``Leaderboard`` is reached through the package at all; every other
consumer in ``src/`` and ``tests/`` imports by submodule path (``from mlframe.votenrank.rank_splice import
segment_rank_splice``, and so on). Nothing is removed, because a name that is unused inside this repository may
still be someone's public entry point: each is resolved on first attribute access instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# name -> submodule it lives in. Keep sorted by submodule for readability; `__all__` below is derived from it,
# so a name added here is exported automatically and cannot drift out of sync with the declared surface.
_LAZY_EXPORTS: dict[str, str] = {
    "compute_test_likeness": "adversarial_stochastic_blend",
    "adversarial_stochastic_blend": "adversarial_stochastic_blend",
    "confidence_gated_blend": "confidence_gated_blend",
    "constrained_weight_blend": "constrained_weight_blend",
    "diversity_ablation_report": "correlation_diversity_ablation",
    "recommend_diversity_additions": "correlation_diversity_ablation",
    "dual_optimizer_weight_blend": "dual_optimizer_blend",
    "geometric_weight_blend": "geometric_weight_blend",
    "hill_climb_ensemble": "hill_climb",
    "KNNFallbackPredictor": "knn_fallback_predictor",
    "Leaderboard": "leaderboard",
    "rank_percentile_transform": "rank_percentile_stacking",
    "segment_rank_splice": "rank_splice",
    "shapley_model_values": "shapley_blend",
    "shapley_blend": "shapley_blend",
    "SimilarityBlendEnsemble": "similarity_blend",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Resolve a re-exported name on first access, importing only the submodule that defines it."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{target}", __name__), name)
    globals()[name] = value  # cache, so later accesses skip this hook entirely
    return value


# Five exported callables share a name with the submodule that defines them (`shapley_blend`, and the four
# other `*_blend` entry points). PEP 562's `__getattr__` is consulted ONLY when the attribute is absent -- and
# importing `mlframe.votenrank.shapley_blend` anywhere sets that submodule as an attribute of this package, so
# a later `from mlframe.votenrank import shapley_blend` then yields the MODULE instead of the function, purely
# because something else imported first. That made the resolved object depend on unrelated import order.
# Binding these eagerly costs the same import the caller was about to trigger anyway and makes the answer
# deterministic; every non-colliding name stays lazy.
_SHADOWED_BY_SUBMODULE = tuple(name for name, target in _LAZY_EXPORTS.items() if name == target)
for _name in _SHADOWED_BY_SUBMODULE:
    globals()[_name] = __getattr__(_name)
del _name


def __dir__() -> list:
    """Include the lazy names in `dir()` / tab-completion, which `__getattr__` alone does not."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point of the lazy hook above
    from .adversarial_stochastic_blend import adversarial_stochastic_blend, compute_test_likeness
    from .confidence_gated_blend import confidence_gated_blend
    from .constrained_weight_blend import constrained_weight_blend
    from .correlation_diversity_ablation import diversity_ablation_report, recommend_diversity_additions
    from .dual_optimizer_blend import dual_optimizer_weight_blend
    from .geometric_weight_blend import geometric_weight_blend
    from .hill_climb import hill_climb_ensemble
    from .knn_fallback_predictor import KNNFallbackPredictor
    from .leaderboard import Leaderboard
    from .rank_percentile_stacking import rank_percentile_transform
    from .rank_splice import segment_rank_splice
    from .shapley_blend import shapley_blend, shapley_model_values
    from .similarity_blend import SimilarityBlendEnsemble
