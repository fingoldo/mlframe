"""RFECV prescreen: the univariate / mRMR / callable candidate filter and its nested per-fold precompute.

Carved out of ``_fit.py`` so the fit facade stays under its LOC budget. ``_fit`` re-imports these names, so
``_fit._apply_prescreen`` and the other symbols keep resolving for existing callers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _apply_prescreen(self, *, X, y, candidate_features, verbose):
    """L7: run the configured prescreen and return the filtered candidate-feature list.

    Supported prescreen values:
      - 'univariate_ht' : in-tree Mann-Whitney / Kruskal-Wallis / Kendall / chi-squared + BY-FDR (numba-compiled).
      - callable(X, y) -> list : user-supplied prescreen returning the kept feature names/indices.
    Returns the kept feature names. Falls back to ``candidate_features`` (no-op) on error.
    """
    _p = self.prescreen
    _topk = getattr(self, "prescreen_top_k", None)
    if callable(_p):
        try:
            _kept = list(_p(X[candidate_features] if isinstance(X, pd.DataFrame) else X, y))
        except Exception as exc:
            if verbose:
                logger.warning("Prescreen callable failed (%s); skipping prescreen.", exc)
            else:
                logger.debug("Prescreen callable failed (%s); skipping prescreen.", exc)
            return candidate_features
        if _topk is not None and len(_kept) > _topk:
            _kept = _kept[: int(_topk)]
        return _kept
    # mRMR pre-screening (TODO from pre-Wave-1 high-priority):
    # use the existing ``mlframe.feature_selection.filters.mrmr.MRMR`` to
    # pick the top-K features by min-redundancy / max-relevance MI ranking
    # before the RFECV outer loop. Best when p >> n (e.g. p >= 5000).
    # Cost: O(p^2) MRMR vs O(p * iter) backward elimination - net win
    # when p >= 5000.
    if isinstance(_p, str) and _p.lower() == "mrmr":
        try:
            from mlframe.feature_selection.filters.mrmr import MRMR
        except ImportError as exc:
            if verbose:
                logger.warning("prescreen='mrmr' could not import MRMR (%s); skipping.", exc)
            return candidate_features
        try:
            _Xsub = X[candidate_features] if isinstance(X, pd.DataFrame) else X
            _topk_arg = int(_topk) if _topk else min(500, max(1, len(candidate_features) // 2))
            _mrmr = MRMR(
                full_npermutations=3, cv=3,
                run_additional_rfecv_minutes=False,
                random_seed=getattr(self, "random_state", 0) or 0,
                min_features_fallback=max(1, _topk_arg // 4),
            )
            _mrmr.fit(_Xsub, y)
            _kept = list(_mrmr.get_feature_names_out())
            if _topk_arg and len(_kept) > _topk_arg:
                _kept = _kept[:_topk_arg]
        except Exception as exc:
            if verbose:
                logger.warning("mRMR prescreen failed (%s); skipping.", exc)
            else:
                logger.debug("mRMR prescreen failed (%s); skipping.", exc)
            return candidate_features
        _kept_set = set(_kept)
        return [c for c in candidate_features if c in _kept_set]

    # L7 native impl: in-tree univariate hypothesis-test
    # relevance scoring with BY-FDR correction. Backend selection per
    # (feature, target) dtype:
    #   binary target  + numeric  -> Mann-Whitney U
    #   multiclass     + numeric  -> Kruskal-Wallis H
    #   continuous     + numeric  -> Kendall tau-b
    #   any            + categorical -> Chi-squared independence
    # Numba-compiled rank / U / H / tau kernels (cache=True). No external
    # dependencies.
    if isinstance(_p, str) and _p.lower() == "univariate_ht":
        if not isinstance(X, pd.DataFrame):
            if verbose:
                logger.warning("Prescreen='univariate_ht' needs pandas DataFrame; skipping.")
            return candidate_features
        from .._univariate_ht import calculate_relevance_table as _crt
        try:
            _Xsub = X[candidate_features]
            _rel = _crt(_Xsub, np.asarray(y), fdr_level=float(getattr(self, "prescreen_fdr_level", 0.05)), ml_task="auto", n_jobs=1)
            _rel = _rel[_rel["relevant"]].sort_values("p_value", ascending=True)
            _kept = list(_rel["feature"])
        except Exception as exc:
            if verbose:
                logger.warning("univariate_ht prescreen failed (%s); skipping.", exc)
            else:
                logger.debug("univariate_ht prescreen failed (%s); skipping.", exc)
            return candidate_features
        if _topk is not None and len(_kept) > _topk:
            _kept = _kept[: int(_topk)]
        _kept_set = set(_kept)
        return [c for c in candidate_features if c in _kept_set]
    if verbose:
        logger.warning("Unknown prescreen=%r; skipping.", _p)
    return candidate_features


def _fold_universe_key(train_index) -> int:
    """Stable key for a fold's train rows: hash of the contiguous int64 index bytes. Deterministic splitters yield
    the same train_index on every ``cv.split`` call, so this matches between the upfront precompute and the fold
    loop regardless of fold ORDER."""
    return hash(np.ascontiguousarray(np.asarray(train_index), dtype=np.int64).tobytes())


def _precompute_prescreen_fold_universes(self, *, X, y, groups, cv, full_features, verbose):
    """Run the configured prescreen on EACH fold's train rows only (over the full pre-prescreen universe), returning
    ``{train_index_key: frozenset(kept_features)}``. Computed once (n_folds prescreen passes) and reused across every
    outer iteration since the splits are fixed. Returns None on any failure so scoring falls back to the global
    (in-universe) estimate rather than crashing."""
    universes: dict = {}
    y_arr = np.asarray(y)
    try:
        for _tr_idx, _te_idx in cv.split(X=X, y=y, groups=groups):
            key = _fold_universe_key(_tr_idx)
            if key in universes:
                continue
            X_tr = X.iloc[_tr_idx]
            y_tr = y_arr[_tr_idx]
            kept = _apply_prescreen(self, X=X_tr, y=y_tr, candidate_features=full_features, verbose=False)
            universes[key] = frozenset(kept)
    except Exception as exc:
        if verbose:
            logger.warning("Nested prescreen precompute failed (%s); using the global in-universe estimate.", exc)
        else:
            logger.debug("Nested prescreen precompute failed (%s); using the global in-universe estimate.", exc)
        return None
    if verbose:
        logger.info("RFECV nested prescreen: precomputed train-only universes for %d fold(s).", len(universes))
    return universes
