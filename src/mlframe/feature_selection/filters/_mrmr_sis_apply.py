"""Layer A-gate — SIS front-screen application for ``MRMR`` (carved out of the class body).

``MRMR._apply_sis_screen(X, y)`` is implemented here as a module-level function taking ``self`` as its
first argument and bound onto the class in ``mrmr/__init__.py`` the same way ``_fit_impl`` / ``_run_fe_step``
are. Living in a sibling shrinks the LOC-exempt ``_mrmr_class.py`` estimator class body. The body is moved
VERBATIM (no logic / threshold / RNG change); only the leading indentation level was removed. The downstream
selection path is byte-for-byte identical - this is pure I/O + column subsetting around the standalone
``sis_screen`` kernel, not a kernel rewrite.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _apply_sis_screen(self, X, y):
    """Gate A: run the chunked SIS screen and subset ``X`` to the survivor columns.

    Returns a column-subset of ``X`` (same container type: pandas / polars / numpy). Reuses the
    standalone ``sis_screen`` (filters/_mrmr_sis_screen) - no kernel logic here, just I/O + subsetting.
    ``k_target`` feeds the survivor floor (the requested number of finally-selected features)."""
    from ._mrmr_sis_screen import sis_screen  # filters-level sibling (was ``from .._mrmr_sis_screen`` in the class body)

    # ndarray view of X for the screen (the screen reads it in column blocks; a memmap stays on disk).
    # NON-NUMERIC SAFETY: the screen casts each column block to float32, so any
    # string/object/categorical column would raise and the outer try would SILENTLY fall back to full-width
    # MRMR (defeating the gate). Factorise non-numeric columns to integer codes here so categoricals are
    # SCORED by the marginal-MI channel (codes are valid MI input) rather than crashing. Numeric columns
    # pass through unchanged (float). The 2nd-moment channel will score nominal codes ~uninformatively, which
    # is acceptable - categorical interaction is not what that channel targets.
    def _numeric_matrix(df):
        """Coerce ``df`` to an all-float ndarray for the SIS scoring channels: numeric/bool-excluded columns pass through as float64, non-numeric columns are factorized to integer codes so they're scored (not silently dropped or crashed on)."""
        cols_out = []
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s.dtype) and s.dtype != bool:
                cols_out.append(np.asarray(s.to_numpy(), dtype=np.float64))
            else:
                cols_out.append(pd.factorize(s, sort=False)[0].astype(np.float64))
        return np.column_stack(cols_out) if cols_out else np.empty((len(df), 0))

    if isinstance(X, pd.DataFrame):
        Xmat = X.to_numpy()
        if Xmat.dtype.kind in "USO" or Xmat.dtype == object:  # mixed/object frame -> factorise per column
            Xmat = _numeric_matrix(X)
    elif str(type(X).__module__).startswith("polars"):
        Xmat = X.to_numpy()
        if Xmat.dtype.kind in "USO" or Xmat.dtype == object:
            Xmat = _numeric_matrix(X.to_pandas())
    else:
        Xmat = np.asarray(X)
        if Xmat.dtype.kind in "USO" or Xmat.dtype == object:  # object ndarray -> factorise per column
            Xmat = _numeric_matrix(pd.DataFrame(Xmat))

    # MRMR has no requested feature count (the greedy stops on its own information criterion), so there is no target to scale the
    # survivor floor by; the screen's absolute floor applies. ``k_target`` stays on ``sis_screen`` for direct callers that do have one.
    k_target = None

    # return_scores=True is FREE (the scores are already computed for survivor selection). We STASH the
    # survivor marginal-MI as a relevance prior so the screen's most expensive output is no longer discarded
    # (reuse audit RU-2). NB it is NOT fed into screen_predictors' cached_MIs as a warm-start: SIS bins
    # quantile-nbins-10 on RAW columns BEFORE categorize, whereas screen_predictors scores MI on categorize's
    # (default MDLP) codes - the two MI values differ, so substituting would CHANGE selection. The recompute
    # it would save is ~3.6s at 100k (CK kernel audit) - second-order vs the ~290s Fleuret CMI loop (CK-1) -
    # so the cached_MIs warm-start is deferred behind CK-1 rather than destabilising the 900-line screen for
    # ~1%. The prior is exposed for diagnostics / a future binning-aligned warm-start.
    survivors, _sis_scores = sis_screen(
        Xmat, y, k_target=k_target,
        dedup_corr_thr=float(getattr(self, "sis_dedup_corr_thr", 0.92) or 0.0),
        return_scores=True,
    )
    survivors = np.asarray(survivors, dtype=np.int64)
    _mi_full = _sis_scores.get("mi")
    self.sis_relevance_prior_ = {int(s): float(_mi_full[int(s)]) for s in survivors} if _mi_full is not None else {}
    self.sis_survivors_ = survivors
    self.sis_n_input_features_ = int(Xmat.shape[1])
    logger.info(
        "[MRMR] SIS front gate: %d -> %d survivors (k_target=%s)",
        int(Xmat.shape[1]), int(survivors.size), k_target,
    )

    # Subset in the caller's container type so the downstream path is identical to a natively narrow frame.
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, survivors]
    if str(type(X).__module__).startswith("polars"):
        return X[:, survivors.tolist()]
    # An ndarray has no names, and the fit would synthesize ``feature_<j>`` from SUBSET positions, so "feature_3" would mean the 4th
    # survivor. Name each survivor by its INPUT position instead, so every name, recipe source and ``support_`` entry can be mapped back.
    return pd.DataFrame(X[:, survivors], columns=[f"feature_{int(i)}" for i in survivors])


def _sis_input_space(X) -> tuple:
    """``(names, synthesized)`` describing the caller's full input, captured before the SIS gate narrows it."""
    if hasattr(X, "columns"):
        return list(X.columns), False
    return [f"feature_{i}" for i in range(int(X.shape[1]))], True


def _remap_sis_fit_to_input_space(self, full_names, synthesized) -> None:
    """Re-express a fit made on the SIS survivors in the caller's input space.

    The fit saw only the survivors, so ``feature_names_in_`` / ``n_features_in_`` / ``support_`` describe that subset. ``transform`` checks
    an ndarray's width against ``n_features_in_`` and indexes it positionally with ``support_``, so both must refer to the full input. The
    mapping goes through names, which are unique and identical in both spaces.
    """
    position = {name: i for i, name in enumerate(full_names)}
    fitted_names = list(self.feature_names_in_)
    support = np.asarray(self.support_)
    if support.dtype == bool:
        support = np.flatnonzero(support)
    self.support_ = np.asarray([position[fitted_names[int(i)]] for i in support], dtype=np.int64)
    self.feature_names_in_ = np.asarray(full_names, dtype=object)
    self.n_features_in_ = len(full_names)
    self._feature_names_in_synthesized_ = bool(synthesized)
