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

import copy
import gc
import hashlib
import logging
import math
import os
import textwrap
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import combinations, islice
from timeit import default_timer as timer
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import make_scorer

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

"""Small top-level helpers for MRMR._fit_impl.

Carved verbatim out of the former flat _mrmr_fit_impl.py when it was
promoted to a subpackage. The single giant _fit_impl body lives in the
sibling _fit_impl_core.py; these are the few small free functions it and
external callers (LRU byte-cap gate, orth-FE scorer dispatch) rely on.
"""


# 2026-06-01 Layer 85 — default-scorer dispatcher for the Layer 21 hybrid
# orth-poly univariate basis-selection stage. Imports the alternate scorer
# module lazily so callers paying for "plug_in" (the default) do not pay
# the import cost. Every alternate scorer's ``*_with_recipes`` returns the
# same ``(X_aug, scores_df, recipes_list)`` 3-tuple as Layer 21's plain
# univariate path -- so the caller plumbing in ``_fit_impl`` is unchanged.
def _orth_fe_numeric_cols(X, cols):
    """Keep only numeric (incl. bool) scalar columns from ``cols`` for the orthogonal / polynomial hybrid-FE family,
    which converts operands to float. Raw categorical / string columns (e.g. a string-coded cat 'B') would otherwise
    raise ``ValueError: could not convert string to float`` and the whole FE pass would be silently dropped. Duplicated
    column names (``X[c]`` -> DataFrame, ndim 2) are skipped as ambiguous."""
    out = []
    for c in cols:
        if c not in X.columns:
            continue
        s = X[c]
        if getattr(s, "ndim", 1) != 1:
            continue
        # pd.api.types.is_numeric_dtype handles category / object / string gracefully (False) and keeps bool as numeric;
        # np.issubdtype would RAISE "Cannot interpret CategoricalDtype as a data type" on the very cat cols we exclude.
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
    return out


def _dispatch_default_scorer(
    scorer: str,
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    cols,
    degrees,
    basis: str,
    top_k: int,
):
    """Route a non-``plug_in`` ``fe_hybrid_orth_default_scorer`` value to
    the matching ``*_with_recipes`` univariate-stage builder.

    Returns the same ``(X_aug, scores_df, recipes_list)`` tuple every
    alternate scorer's ``with_recipes`` variant exposes. Raises
    ``ValueError`` on an unrecognised scorer string (defensive: the public
    validation entry point catches this in ``_validate_string_params``).
    """
    if scorer == "cmim":
        from .._orthogonal_cmim_fe import (
            hybrid_orth_mi_cmim_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "jmim":
        from .._orthogonal_jmim_fe import (
            hybrid_orth_mi_jmim_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "tc":
        from .._orthogonal_total_correlation_fe import (
            hybrid_orth_mi_tc_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "ksg":
        from .._orthogonal_ksg_mi_fe import (
            hybrid_orth_mi_ksg_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "copula":
        from .._orthogonal_copula_mi_fe import (
            hybrid_orth_mi_copula_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "dcor":
        from .._orthogonal_dcor_fe import (
            hybrid_orth_mi_dcor_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "hsic":
        from .._orthogonal_hsic_fe import (
            hybrid_orth_mi_hsic_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "lasso":
        from .._orthogonal_lasso_fe import (
            hybrid_orth_mi_lasso_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "elasticnet":
        from .._orthogonal_elasticnet_fe import (
            hybrid_orth_mi_elasticnet_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "auto":
        from .._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_auto_scorer_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "ensemble":
        from .._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_ensemble_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "meta":
        from .._orthogonal_meta_scorer_fe import (
            hybrid_orth_mi_meta_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "auto_oracle":
        # 2026-06-01 Layer 100 — UNIFIED scorer-selection. The
        # OracleScorerSelector resolves a concrete scorer for this
        # dataset's fingerprint (learned-best when the oracle has confident
        # history, else the L76 cold-start cascade), then we delegate to
        # THAT scorer's univariate builder. The bake-off that populates the
        # oracle (``benchmark_all_scorers``) runs out-of-band, so fit-time
        # cost is one recommend + one scorer run, not the full L68 sweep.
        from .._oracle_scorer_select import OracleScorerSelector
        _selector = OracleScorerSelector()
        _resolved = _selector.recommend_scorer(X, y)
        # ``auto_oracle`` never resolves to itself / "plug_in"-by-default is
        # routed through the plug_in builders below; any other resolved
        # scorer recurses into its own branch of this dispatcher.
        if _resolved == "plug_in":
            from .._orthogonal_univariate_fe import (
                hybrid_orth_mi_fe_with_recipes as _fn,
            )
            return _fn(
                X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k,
            )
        return _dispatch_default_scorer(
            _resolved, X=X, y=y, cols=cols, degrees=degrees,
            basis=basis, top_k=top_k,
        )
    raise ValueError(
        f"_dispatch_default_scorer: unrecognised scorer={scorer!r}. "
        f"This is a defensive bug -- ``_validate_string_params`` should "
        f"have caught the bad value before fit reached the dispatcher."
    )


def _mrmr_instance_state_size_bytes(instance: Any) -> int:
    """Best-effort byte estimate for a single fitted MRMR instance's selector + engineered-features state.

    Used by the LRU eviction byte gate. Walks the small set of large state attributes (``mi_scores_``, ``_selectors_``, ``_engineered_features_``, ``_y_full_hash`` retained y) so the estimate reflects the dominant footprint without paying ``pickle.dumps`` cost on every eviction probe.
    """
    total = 0
    for _attr in ("mi_scores_", "_selectors_", "_engineered_features_", "ranking_", "support_", "selected_features_"):
        try:
            _v = getattr(instance, _attr, None)
            if _v is None:
                continue
            _nb = getattr(_v, "nbytes", None)
            if isinstance(_nb, int):
                total += _nb
                continue
            if isinstance(_v, dict):
                for _vv in _v.values():
                    _vvnb = getattr(_vv, "nbytes", None)
                    if isinstance(_vvnb, int):
                        total += _vvnb
                    else:
                        try:
                            total += int(np.asarray(_vv).nbytes)
                        except Exception:
                            pass
            elif isinstance(_v, (list, tuple)):
                for _item in _v:
                    _inb = getattr(_item, "nbytes", None)
                    if isinstance(_inb, int):
                        total += _inb
        except Exception:
            continue
    return total


def _mrmr_cache_bytes_total() -> int:
    """Sum of state bytes across every cached MRMR instance in MRMR._FIT_CACHE."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    return sum(_mrmr_instance_state_size_bytes(_v) for _v in MRMR._FIT_CACHE.values())