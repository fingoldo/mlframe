"""``MRMR._fit_impl`` main fit body for ``mlframe.feature_selection.filters.mrmr``.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. ``_fit_impl`` is bound back onto the ``MRMR`` class at the
parent's module bottom, so call sites that invoke ``self._fit_impl(...)``
continue to work unchanged.

Heavy lifting: signature/cache key build, content-hash short-circuit,
sub-sample loop, FE-step orchestration, MI ranking and the per-fold
selection. Many helpers (logger, signature hashing, target coercion)
live in the parent and are imported lazily inside this body to avoid the
``mrmr -> _mrmr_fit_impl -> mrmr`` import cycle.
"""

from __future__ import annotations

import logging
import threading


import numpy as np

import pandas as pd
from mlframe.utils.log_throttle import log_throttle  # noqa: F401 - kept as a facade name (moved callers live in _fit_impl_gains)

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Guards every read-then-mutate sequence on the process-wide ``MRMR._FIT_CACHE`` (lookup + ``move_to_end`` on
# a hit; ``__setitem__`` + ``move_to_end`` + LRU/byte-cap ``popitem`` on store). Concurrent fits - multi-target
# discovery, joblib-threading callers, web-service workers - otherwise race ``popitem``/``__setitem__``/
# ``move_to_end`` on the same OrderedDict and can raise KeyError or evict the wrong entry. RLock so a wrapped
# region may safely re-enter. The companion ``_MRMR_IDENTITY_FP_CACHE`` already had its own lock; this closes the
# same gap for the fit cache. Exposed on the ``MRMR`` class (idempotently, inside the fit body) as
# ``_FIT_CACHE_LOCK`` so any other holder of the cache can take the same canonical lock.
_MRMR_FIT_CACHE_LOCK = threading.RLock()


# _pgn_raw_budget re-exported from the ``_assign_support`` sub-split (Tier F) for backward-compat
# direct imports (e.g. tests/feature_selection/filters/test_mrmr_pgn_engineered_budget.py) that
# import it from this module's own namespace.
from ._assign_support import _pgn_raw_budget  # noqa: F401 - re-exported facade name, imported directly by tests/test_mrmr_pgn_engineered_budget.py

# Above this many bytes of nullable-column data, densify masked columns one-per-``assign`` instead of all at once
# so peak extra RAM stays ~one float64 column rather than ~2x the whole nullable subset (100GB-frame safe).
_NULLABLE_DENSIFY_EAGER_MAX_BYTES = 2 * 1024**3


from ._fit_impl_gains import _align_mrmr_gains, _gains_paired_with_output, _resolve_screen_shared_idx  # noqa: F401 - re-exported facade names (tests import _align_mrmr_gains)

"""MRMR._fit_impl main fit body.

The irreducible single function _fit_impl (bound onto the MRMR class
at the mrmr package facade) lives here verbatim. It is LOC-budget exempt:
one giant function cannot be split without distorting the fit control flow.
Its many lazy in-body from ..X import ... imports break the
mrmr -> _mrmr_fit_impl -> mrmr cycle; the small free helpers it calls live
in the sibling _helpers.py.
"""


from ._helpers import _mrmr_cache_bytes_total
from ._fit_impl_core_parts import _fit_impl_part7, _fit_impl_part9


def _fit_impl(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | pd.Series | np.ndarray, groups: pd.Series | np.ndarray = None, **fit_params):
    """We run N selections on data subsets, and pick only features that appear in all selections"""
    _ret_fit_impl_part7 = _fit_impl_part7(
        y, self, X, groups,
    )
    _ret_fit_impl_part9 = _fit_impl_part9(
        _ret_fit_impl_part7, self, y,
    )
    if _ret_fit_impl_part9[0]:
        return _ret_fit_impl_part9[1]
    MRMR, _hashable_params_signature, recipes, fe, _y_np, X, signature, _cache_key, start_time, ran_out_of_time, _fe_family_on, verbose, cols, data, nbins, target_indices, categorical_vars_names, engineered_recipes, _effective_min_relevance_gain, _persisted_dcd_state, selected_vars, predictors, cached_MIs, classes_y, _eng_continuous_snapshot, _friend_graph_analysis = _ret_fit_impl_part9[1:]

    cols, data, nbins, selected_vars = _friend_graph_analysis(
        self,
        X,
        classes_y,
        cols,
        data,
        nbins,
        target_indices,
        y,
        verbose,
        cached_MIs,
        engineered_recipes,
        _eng_continuous_snapshot,
        selected_vars,
        _effective_min_relevance_gain,
        recipes,
        _persisted_dcd_state,
        _y_np,
        _fe_family_on,
    )
    # ---------------------------------------------------------------------------------------------------------------
    # selected_vars: cols-indices -> names -> original-frame indices (categorize_dataset may rearrange cat columns).
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _map_selected_to_original_indices

    _fni_idx, selected_vars = _map_selected_to_original_indices(self, cols, selected_vars, verbose, engineered_recipes)

    # ---------------------------------------------------------------------------------------------------------------
    # additional_rfecv run
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _run_additional_rfecv

    _run_additional_rfecv(self, X, selected_vars, verbose, y, categorical_vars_names, _fni_idx)

    # ---------------------------------------------------------------------------------------------------------------
    # Assign support
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _assign_support

    _assign_support(
        self,
        X,
        classes_y,
        cols,
        data,
        nbins,
        target_indices,
        y,
        verbose,
        fe,
        cached_MIs,
        engineered_recipes,
        predictors,
        _eng_continuous_snapshot,
        selected_vars,
    )

    # ---------------------------------------------------------------------------------------------------------------
    # Report FS results
    # ---------------------------------------------------------------------------------------------------------------
    from ._finalise import _finalise_fs_results

    return _finalise_fs_results(
        self,
        MRMR=MRMR,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        predictors=predictors,
        start_time=start_time,
        verbose=verbose,
        cache_key=_cache_key,
        signature=signature,
        ran_out_of_time=ran_out_of_time,
        hashable_params_signature=_hashable_params_signature,
        mrmr_cache_bytes_total=_mrmr_cache_bytes_total,
        align_mrmr_gains=_align_mrmr_gains,
        fit_cache_lock=_MRMR_FIT_CACHE_LOCK,
    )
