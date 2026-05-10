"""mRMR feature-selection package.

Public API
----------
``MRMR`` is the sklearn-compatible estimator. The helper symbols
(``entropy``, ``mi``, ``conditional_mi``, ``merge_vars``,
``compute_mi_from_classes``, ``categorize_dataset``, ``discretize_array``,
and friends) are re-exported here for convenience and BC -- importers in
``mlframe/training/*``, ``mlframe/finance/*``, and the test suite use this
top-level module.

Where does new code go?
-----------------------
1. Shared ``@njit`` helper used from > 1 submodule -> ``_numba_utils.py``
2. Public mRMR algorithmic phase -> ``screen.py`` (screening) or ``mrmr.py``
3. Primitive math op (entropy/MI/discretization) -> ``info_theory.py`` /
   ``discretization.py``
4. GPU-specific (CuPy/CUDA) -> ``gpu.py``
5. Permutation / confidence test -> ``permutation.py`` / ``fleuret.py``
6. Single-module ``@njit`` helper -> the owning module
7. Constants -> ``_internals.py`` with a docstring explaining what the
   bound prevents, the failure mode if exceeded, and the empirical /
   algorithmic basis for the value chosen.

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

History
-------
This used to be a 4187-LOC monolith ``filters.py`` (deleted at end of the
refactor; for the in-progress migration, the legacy code lives in
``_legacy.py`` and submodules siphon symbols out one etap at a time).
"""

# During the move-and-fix etapes (1-10) the legacy monolith is the source of
# truth and submodule files siphon symbols out of it. Re-exporting `*` keeps
# every existing importer working transparently.
from ._legacy import *  # noqa: F401,F403
from ._legacy import (  # explicit -- kept stable across migration etapes
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

# B27 placeholder: pickle BC for the upcoming `max_confirmation_cand_nbins`
# default change (B13). Once etap 11 lands the kwarg, ``__setstate__`` on the
# class will inject the legacy default for old pickles. Until then the
# attribute is module-level constant and old pickles still resolve fine.

__all__ = [
    "MRMR",
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
]


# Pickle BC: MRMR was historically saved with __module__="mlframe.feature_selection.filters".
# After the move into a package, the class object's __module__ becomes "..._legacy"
# (or "..._mrmr" once etap 11 lands). Reassign here so cloudpickle / joblib continue to
# resolve unpickling against the package's __init__ where MRMR is re-exported.
MRMR.__module__ = __name__
for _attr in ("fit", "transform", "fit_transform"):
    _f = getattr(MRMR, _attr, None)
    if _f is not None and hasattr(_f, "__module__"):
        try:
            _f.__module__ = __name__
        except (AttributeError, TypeError):
            pass
del _attr


# B13 (post-plan): the legacy ``MAX_CONFIRMATION_CAND_NBINS`` module
# constant is replaced by the per-instance ``MRMR(max_confirmation_cand_nbins=...)``
# kwarg with a smarter default (``nbins ** order * 2``). For one release
# we keep the constant accessible from the package namespace so that
# ad-hoc scripts that did ``from mlframe.feature_selection.filters import
# MAX_CONFIRMATION_CAND_NBINS`` keep working, but emit a ``DeprecationWarning``
# pointing them at the new kwarg.
# Remove from namespace so __getattr__ below traps the access and emits
# the DeprecationWarning. Without this the wildcard import from _legacy
# already populates the name and __getattr__ never fires.
if "MAX_CONFIRMATION_CAND_NBINS" in globals():
    del globals()["MAX_CONFIRMATION_CAND_NBINS"]


def __getattr__(name):
    if name == "MAX_CONFIRMATION_CAND_NBINS":
        import warnings as _w
        from ._internals import MAX_CONFIRMATION_CAND_NBINS as _legacy_const
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

