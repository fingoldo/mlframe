"""mRMR feature-selection package.

Public API
----------
``MRMR`` is the sklearn-compatible estimator. The helper symbols (``entropy``, ``mi``, ``conditional_mi``, ``merge_vars``, ``compute_mi_from_classes``,
``categorize_dataset``, ``discretize_array``, ...) are re-exported here for convenience and BC -- importers in ``mlframe/training/*``, ``mlframe/finance/*``,
and the test suite use this top-level module.

Where does new code go?
-----------------------
1. Shared ``@njit`` helper used from > 1 submodule -> ``_numba_utils.py``
2. Public mRMR algorithmic phase -> ``screen.py`` (screening) or ``mrmr.py``
3. Primitive math op (entropy/MI/discretization) -> ``info_theory.py`` / ``discretization.py``
4. GPU-specific (CuPy/CUDA) -> ``gpu.py``
5. Permutation / confidence test -> ``permutation.py`` / ``fleuret.py``
6. Single-module ``@njit`` helper -> the owning module
7. Constants -> ``_internals.py`` with a docstring explaining what the bound prevents, the failure mode if exceeded, and the empirical / algorithmic basis.

Module dependency graph (acyclic)::

    _internals  <-  _numba_utils  <-  info_theory  <-  permutation  <-  evaluation
        ^                ^               ^               ^                ^
        |                |               |             gpu                |
        |                |               |               ^                |
        |                |               +---------------+----------------+
        |                |                               |             fleuret
        |                |                               |                ^
        |                |                            screen --------------+
        |                |                               ^
    discretization                                    mrmr ---> feature_engineering
"""

from __future__ import annotations


# Legacy monolith star-import keeps every existing importer working transparently.
from ._legacy import *  # noqa: F401,F403
from ._legacy import (  # explicit re-exports, kept stable
    MRMR,
    entropy,
    mi,
    conditional_mi,
    merge_vars,
    compute_mi_from_classes,
    categorize_dataset,
    discretize_array,
    discretize_uniform,
    discretize_2d_array,
    discretize_sklearn,
    get_binning_edges,
    arr2str,
    mi_direct,
    parallel_mi,
    distribute_permutations,
)
from ._mrmr_tree_rescue import MRMRTreeRescued

__all__ = [
    "MRMR",
    "MRMRTreeRescued",
    "entropy",
    "mi",
    "conditional_mi",
    "merge_vars",
    "compute_mi_from_classes",
    "categorize_dataset",
    "discretize_array",
    "discretize_uniform",
    "discretize_2d_array",
    "discretize_sklearn",
    "get_binning_edges",
    # Cat-FE public surface
    "CatFEConfig",
    "CatFEState",
    "EngineeredRecipe",
    "run_cat_interaction_step",
    # JIT pre-warm hooks for production process bootstrap.
    "prewarm_fs_numba_cache",
    "prewarm_fs_cupy_kernels",
    # Kernel-tuning cache accessor (public surface; underscore source remains the implementation).
    "get_kernel_tuning_cache",
    # Cluster-aggregate helpers (used by sibling _shap_proxy_cluster.py to build
    # post-clustering aggregate features under the same numerical contract as the
    # filters._cluster_aggregate path).
    "derive_cluster_weights",
    "standardize_align_cluster",
    "apply_cluster_aggregate_nonlinear",
]


# Pre-warm hooks. Idempotent; call once per process before first MRMR.fit to
# move ~5-15s of numba+CuPy NVRTC compile out of the timed fit path.
from ._prewarm import prewarm_fs_numba_cache, prewarm_fs_cupy_kernels  # noqa: E402
from ._kernel_tuning import get_kernel_tuning_cache  # noqa: E402, F401
from ._cluster_aggregate import (  # noqa: E402, F401
    _derive_weights as derive_cluster_weights,
    _standardize_align as standardize_align_cluster,
    _apply_method_nonlinear as apply_cluster_aggregate_nonlinear,
)


# Cat-FE re-exports. Imported here (not via the ``_legacy`` star) so that
# ``from mlframe.feature_selection.filters import CatFEConfig`` works as advertised in the public ``MRMR(cat_fe_config=CatFEConfig(...))`` API.
from .cat_fe_state import CatFEConfig, CatFEState  # noqa: E402
from .cat_interactions import run_cat_interaction_step  # noqa: E402
from .engineered_recipes import EngineeredRecipe  # noqa: E402


# Pickle BC: MRMR was historically saved with __module__="mlframe.feature_selection.filters". After the move into a package, the class object's __module__
# becomes "..._legacy" (or "..._mrmr"). Reassign here so cloudpickle / joblib continue to resolve unpickling against the package's __init__.
MRMR.__module__ = __name__
for _attr in ("fit", "transform", "fit_transform"):
    _f = getattr(MRMR, _attr, None)
    if _f is not None and hasattr(_f, "__module__"):
        try:
            _f.__module__ = __name__
        except (AttributeError, TypeError):
            pass
del _attr


# Deprecation trap for the legacy ``MAX_CONFIRMATION_CAND_NBINS`` module constant, now replaced by per-instance ``MRMR(max_confirmation_cand_nbins=...)``
# kwarg with default ``nbins ** order * 2``. Remove from namespace so ``__getattr__`` below traps access; otherwise the wildcard import from ``_legacy``
# populates the name and ``__getattr__`` never fires.
if "MAX_CONFIRMATION_CAND_NBINS" in globals():
    del globals()["MAX_CONFIRMATION_CAND_NBINS"]


def __getattr__(name):
    if name == "_hashable_params_signature":
        # Lazy public re-export. ``_mrmr_fingerprints`` top-level imports ``wrappers`` (RFECV),
        # so a module-level import here would close a package-init cycle; resolving it on first
        # access keeps the public path available to out-of-package callers without the cycle.
        from ._mrmr_fingerprints import _hashable_params_signature as _fn
        return _fn
    if name == "MAX_CONFIRMATION_CAND_NBINS":
        import warnings as _w
        from ._internals import MAX_CONFIRMATION_CAND_NBINS as _legacy_const  # noqa: N811 -- deprecation-trap alias, name intentionally non-constant-looking
        _w.warn(
            "Accessing MAX_CONFIRMATION_CAND_NBINS as a module constant is "
            "deprecated. Use MRMR(max_confirmation_cand_nbins=...) per-instance "
            "instead. The legacy constant value (50) is no longer the default; "
            "MRMR.fit now defaults to ``quantization_nbins ** interactions_max_order * 2``.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _legacy_const
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
